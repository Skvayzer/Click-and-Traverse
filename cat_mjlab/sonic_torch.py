"""GPU-batched torch port of the released Default GEAR-SONIC g1 path, with access to the pre-quantization latent.

Weights are read from the pinned release ONNX files (nvidia/GEAR-SONIC@6733128a, default variant):
    encoder g1 branch  MLP 640 -> 2048 -> 1024 -> 512 -> 512 -> 64, SiLU; input = per future frame (10) the 58 joint
                       command values (the release's packing: all positions then all velocities, reshaped to 10 x 58)
                       followed by the 6 anchor-orientation values, flattened to 640
    FSQ                2 tokens x 32 dims, 32 levels: s = tanh(z + 0.032237) * 15.515501 - 0.5; q = round(s) / 16
    decoder g1_dyn     MLP 994 -> 2048 -> 2048 -> 1024 -> 1024 -> 512 -> 512 -> 29, SiLU; input = 64 tokens + 930
                       proprioception
`act(cmd, anchor, proprio, residual=None, scale)` adds `scale * residual` to the 64-D latent BEFORE FSQ (the
play2perfect/GRAIL frozen latent-adaptor design); residual None or 0 reproduces the release exactly. All weights are
frozen (requires_grad False). `verify(...)` compares against the release ONNX graphs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

ENCODER_SHA256 = "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3"
DECODER_SHA256 = "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed"
FSQ_SHIFT, FSQ_HALF, FSQ_OFFSET, FSQ_WIDTH = 0.032237, 15.515501, 0.5, 16.


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _onnx_weights(path, expected, prefix, _transpose=None):
    import onnx
    from onnx import numpy_helper

    if _sha(path) != expected:
        raise ValueError(f"SONIC file differs from the pinned release: {path}")
    model = onnx.load(str(path))
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    layers = []
    for node in model.graph.node:
        if node.op_type == "Gemm" and node.name.startswith(prefix):                    # encoder: W [out, in]
            layers.append((inits[node.input[1]], inits[node.input[2]]))
        elif node.op_type == "MatMul" and node.name.startswith(prefix):                # decoder: W [in, out], Add bias
            add = next(n for n in model.graph.node if n.op_type == "Add" and node.output[0] in n.input)
            bias = next(inits[i] for i in add.input if i in inits)
            layers.append((inits[node.input[1]].T, bias))
    consts = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in model.graph.node if n.op_type == "Constant"}
    for node in model.graph.node:              # the FSQ constants must be the ones this port hard-codes
        if node.name == "/quantizer/Add" and not np.allclose(consts[node.input[1]], FSQ_SHIFT, atol=1e-6):
            raise ValueError("FSQ shift differs")
        if node.name == "/quantizer/Mul" and not np.allclose(consts[node.input[1]], FSQ_HALF, atol=1e-5):
            raise ValueError("FSQ scale differs")
    return layers


class _MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.linears = nn.ModuleList()
        for w, b in layers:
            lin = nn.Linear(w.shape[1], w.shape[0])
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(w)))
                lin.bias.copy_(torch.from_numpy(b))
            self.linears.append(lin)

    def forward(self, x):
        for k, lin in enumerate(self.linears):
            x = lin(x)
            if k < len(self.linears) - 1:
                x = x * torch.sigmoid(x)
        return x


def fsq(z):
    s = torch.tanh(z + FSQ_SHIFT) * FSQ_HALF - FSQ_OFFSET
    return (s + (torch.round(s) - s).detach()) / FSQ_WIDTH      # straight-through; values as in the release


class SonicTorch(nn.Module):
    def __init__(self, encoder_path, decoder_path, device="cuda"):
        super().__init__()
        self.encoder = _MLP(_onnx_weights(encoder_path, ENCODER_SHA256, "/g1/", False))
        self.decoder = _MLP(_onnx_weights(decoder_path, DECODER_SHA256, "/g1_dyn/", True))
        if [l.weight.shape for l in self.encoder.linears][0] != (2048, 640) or self.decoder.linears[-1].weight.shape != (29, 512):
            raise ValueError("Unexpected SONIC layer shapes")
        self.requires_grad_(False)
        self.eval()
        self.to(device)

    def train(self, mode=True):
        return super().train(False)

    def latent(self, cmd, anchor):
        """cmd [N, 580] (release packing), anchor [N, 60] -> pre-FSQ latent [N, 64]."""
        n = cmd.shape[0]
        x = torch.cat((cmd.reshape(n, 10, 58), anchor.reshape(n, 10, 6)), -1).reshape(n, 640)
        return self.encoder(x)

    @torch.no_grad()
    def act(self, cmd, anchor, proprio, residual=None, scale=.1):
        z = self.latent(cmd, anchor)
        if residual is not None:
            z = z + scale * residual
        tokens = fsq(z)
        return self.decoder(torch.cat((tokens, proprio), -1)), z, tokens


def verify(encoder_path, decoder_path, n=64, device="cpu", seed=0):
    """Max |torch - ONNX| for tokens (fraction of differing FSQ codes too) and decoder actions on random inputs."""
    import onnxruntime as ort
    from .sonic_tracker import batched_models

    enc_blob, dec_blob = batched_models(encoder_path, decoder_path)
    enc = ort.InferenceSession(enc_blob, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(dec_blob, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 1762)).astype(np.float32) * .5
    x[:, :4] = 0.
    proprio = rng.standard_normal((n, 930)).astype(np.float32) * .5
    tokens_onnx = enc.run(None, {"obs_dict": x})[0]
    model = SonicTorch(encoder_path, decoder_path, device=device)
    with torch.no_grad():
        t = lambda a: torch.from_numpy(a).to(device)
        z = model.latent(t(x[:, 4:584]), t(x[:, 601:661]))
        tokens = fsq(z).cpu().numpy()
        act_onnx = dec.run(None, {"obs_dict": np.concatenate((tokens_onnx, proprio), -1)})[0]
        act_torch = model.decoder(torch.cat((t(tokens_onnx), t(proprio)), -1)).cpu().numpy()
        act_chain = model.act(t(x[:, 4:584]), t(x[:, 601:661]), t(proprio))[0].cpu().numpy()
        act_zero = model.act(t(x[:, 4:584]), t(x[:, 601:661]), t(proprio), residual=torch.zeros(n, 64, device=device))[0]
    return dict(token_max_abs=float(np.abs(tokens - tokens_onnx).max()),
                token_code_mismatch_fraction=float((np.abs(tokens - tokens_onnx) > 1e-6).mean()),
                decoder_max_abs=float(np.abs(act_torch - act_onnx).max()),
                chain_max_abs=float(np.abs(act_chain - act_onnx).max()),
                zero_residual_identical=bool(np.array_equal(act_zero.cpu().numpy(), act_chain)))
