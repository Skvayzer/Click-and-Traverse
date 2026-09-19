"""Bounded workarounds for the pinned Torch 2.7 CUDA compiler.

Its singleton specialization can generate invalid Triton for concatenated,
broadcast quaternions. Selective one-world resets therefore run eagerly;
larger batches keep the same compiled numerical kernels. Compiler errors in
other shapes propagate instead of silently changing execution mode.
"""
from __future__ import annotations

from functools import wraps
from inspect import signature

import torch


def compile_batched_kernel(function,*,batch_arg=None,backend='inductor'):
    """Compile batches except B=1, using a named/indexed argument if supplied.

    By default select the first direct tensor argument (positional first, then
    keywords). An explicit selector is required when a shared, non-batched
    tensor precedes the batch, as with the immutable field sampler.
    """
    options=dict(backend=backend,fullgraph=True,dynamic=True)
    if backend=='inductor':options['mode']='default'
    compiled=torch.compile(function,**options)
    parameter_names=tuple(signature(function).parameters)
    if isinstance(batch_arg,str):
        if batch_arg not in parameter_names:raise ValueError(f'Unknown batch argument: {batch_arg}')
        position=parameter_names.index(batch_arg)
    else:position=batch_arg

    @wraps(function)
    def dispatch(*args,**kwargs):
        if batch_arg is None:
            batch=next((value for value in (*args,*kwargs.values()) if isinstance(value,torch.Tensor)),None)
        elif isinstance(batch_arg,str):
            batch=kwargs[batch_arg] if batch_arg in kwargs else args[position]
        else:
            batch=args[position] if position<len(args) else kwargs[parameter_names[position]]
        if not isinstance(batch,torch.Tensor) or batch.ndim==0:
            raise ValueError('Compiled task kernel needs a batched tensor argument')
        return (function if batch.shape[0]==1 else compiled)(*args,**kwargs)

    return dispatch
