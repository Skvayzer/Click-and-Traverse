"""The portable resume entry reconstructs settings and never fresh-starts."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/resume_cat_sapg.py"
SPEC = importlib.util.spec_from_file_location("cat_sapg_resume_entry", SCRIPT)
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = (tmp_path / "pinned").resolve()
    source.mkdir()
    run = (tmp_path / "run").resolve()
    run.mkdir()
    commit = "1" * 40
    code = dict(git_commit=commit, source_sha256="a" * 64)
    config = dict(algorithm="sapg", policy_config=dict(num_envs=36864, batch_size=576, seed=7,
                  learning_rate=.0003, entropy_cost=.003, clipping_epsilon=.2, num_evals=0),
                  fine_tuning=dict(mode="cat_train_only", profile="single_gpu_32gb"),
                  sapg=dict(num_policies=6, embedding_dim=16, prepare_chunk_size=64))
    specification = dict(config=config, bank_manifest=str(tmp_path / "fields/manifest.json"),
                         body_collision_bank=str(tmp_path / "collision/manifest.json"),
                         body_collision_resets=str(tmp_path / "resets/manifest.json"),
                         stop=str(run / "STOP"), task="unchanged")
    destination = dict(entity="skvayzer", project="CAT-wholebody", mode="online")
    spec_hash = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    launch = dict(schema="cat-generalist-launch-v1", specification=specification,
                  specification_sha256=spec_hash, owner="owner", wandb=destination)
    record = dict(specification, code=code, warmstart=dict(original=True), code_migrations=["fixed loss"])
    status = dict(owner="owner", specification_sha256=spec_hash, resume_state_steps=451805184)
    identity = dict(id="f017f302", initialized=True, **destination)
    for name, value in (("run.json", record), ("launch.json", launch), ("status.json", status), ("wandb.json", identity)):
        write_json(run / name, value)
    (run / "resume.msgpack").write_bytes(b"full runtime presence checked; normal launcher restores it")
    (source / "train_cat_wholebody.py").write_text("# pinned launcher\n")
    monkeypatch.setattr(resume.subprocess, "check_output", lambda *args, **kwargs: commit + "\n")
    monkeypatch.setattr(resume.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--resume", action="store_true")
    for key in ("run-dir", "algorithm", "finetuning", "profile", "num-envs", "batch-size", "seed",
                "sapg-num-policies", "sapg-embedding-dim", "bank-manifest", "body-collision-bank",
                "body-collision-resets", "wandb-mode", "wandb-project", "wandb-entity"):
        parser.add_argument("--" + key, required=True)

    def plan(arguments):
        result = copy.deepcopy(specification)
        result["config"]["algorithm"] = arguments.algorithm
        for key in ("num_envs", "batch_size", "seed"):
            result["config"]["policy_config"][key] = int(getattr(arguments, key))
        result["config"]["sapg"].update(num_policies=int(arguments.sapg_num_policies),
                                           embedding_dim=int(arguments.sapg_embedding_dim))
        result["config"]["fine_tuning"].update(mode=arguments.finetuning, profile=arguments.profile)
        for key in ("bank_manifest", "body_collision_bank", "body_collision_resets"):
            result[key] = getattr(arguments, key)
        result["stop"] = str(Path(arguments.run_dir) / "STOP")
        return result

    launcher = SimpleNamespace(__file__=str(source / "train_cat_wholebody.py"),
                               code_identity=lambda: code, parser=lambda: parser, plan=plan)
    kwargs = dict(source_root=source, run_dir=run, expected_commit=commit,
                  executable="/verified/venv/bin/python", launcher=launcher)
    return dict(kwargs=kwargs, run=run, source=source, launcher=launcher, launch=launch)


def test_recovers_exact_resources_algorithm_and_original_wandb(setup):
    result = resume.prepare_command(**setup["kwargs"])
    command = result["command"]
    parsed = setup["launcher"].parser().parse_args(command[2:])
    assert parsed.command == "run" and parsed.resume
    assert parsed.algorithm == "sapg" and parsed.finetuning == "cat_train_only"
    assert parsed.num_envs == "36864" and parsed.batch_size == "576" and parsed.seed == "7"
    assert parsed.sapg_num_policies == "6" and parsed.sapg_embedding_dim == "16"
    assert parsed.wandb_entity == "skvayzer" and parsed.wandb_project == "CAT-wholebody"
    assert result["wandb_id"] == "f017f302" and result["saved_step"] == 451805184
    assert command[:2] == ["/verified/venv/bin/python", str(setup["source"] / "train_cat_wholebody.py")]
    assert setup["launcher"].plan(parsed) == setup["launch"]["specification"]
    assert "--warmstart-best" not in command and "--reuse-untrained-wandb-from" not in command


@pytest.mark.parametrize("problem", ["stop", "runtime_missing", "runtime_empty", "runtime_symlink", "branch",
                                    "commit", "code", "wandb", "record", "spec_hash", "plan"])
def test_refuses_inexact_or_unsafe_resume(setup, monkeypatch, problem):
    if problem == "stop":
        (setup["run"] / "STOP").touch()
    elif problem == "runtime_missing":
        (setup["run"] / "resume.msgpack").unlink()
    elif problem == "runtime_empty":
        (setup["run"] / "resume.msgpack").write_bytes(b"")
    elif problem == "runtime_symlink":
        runtime = setup["run"] / "resume.msgpack"
        target = setup["run"] / "other.msgpack"
        runtime.rename(target)
        runtime.symlink_to(target)
    elif problem == "branch":
        monkeypatch.setattr(resume.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    elif problem == "commit":
        setup["kwargs"]["expected_commit"] = "2" * 40
    elif problem == "code":
        setup["launcher"].code_identity = lambda: dict(git_commit=setup["kwargs"]["expected_commit"], source_sha256="b" * 64)
    elif problem == "plan":
        original_plan = setup["launcher"].plan
        setup["launcher"].plan = lambda args: dict(original_plan(args), task="changed")
    else:
        name = {"wandb": "wandb.json", "record": "run.json", "spec_hash": "launch.json"}[problem]
        path = setup["run"] / name
        value = json.loads(path.read_text())
        if problem == "wandb":
            value["project"] = "other"
        elif problem == "record":
            value["config"]["policy_config"]["learning_rate"] = .1
        else:
            value["specification_sha256"] = "0" * 64
        write_json(path, value)
    with pytest.raises(ValueError):
        resume.prepare_command(**setup["kwargs"])


def test_print_command_validates_without_exec_or_environment_changes(setup, monkeypatch, capsys):
    result = resume.prepare_command(**setup["kwargs"])
    monkeypatch.setattr(resume, "prepare_command", lambda *args, **kwargs: result)
    monkeypatch.setattr(resume.os, "execv", lambda *args: pytest.fail("print-command must never run a learner"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "allocated-by-owner")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", ".90")
    before = dict(resume.os.environ)
    resume.main(["--source-root", str(setup["source"]), "--run-dir", str(setup["run"]),
                 "--expected-commit", setup["kwargs"]["expected_commit"], "--print-command"])
    assert "f017f302" in capsys.readouterr().out
    assert dict(resume.os.environ) == before


def test_exec_uses_verified_interpreter_and_pinned_working_directory(setup, monkeypatch):
    result = resume.prepare_command(**setup["kwargs"])
    monkeypatch.setattr(resume, "prepare_command", lambda *args, **kwargs: result)
    calls = []
    monkeypatch.setattr(resume.os, "chdir", lambda path: calls.append(("chdir", path)))
    monkeypatch.setattr(resume.os, "execv", lambda executable, args: calls.append(("exec", executable, args)))
    resume.main(["--source-root", str(setup["source"]), "--run-dir", str(setup["run"]),
                 "--expected-commit", setup["kwargs"]["expected_commit"]])
    assert calls == [("chdir", str(setup["source"])), ("exec", result["command"][0], result["command"])]
