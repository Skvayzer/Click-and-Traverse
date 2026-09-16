"""Pass immutable scene fields as runtime operands instead of large HLO literals."""

from contextlib import contextmanager


class FieldArguments:
    """Trace-local binding for the existing environment's field-reading methods.

    JAX 0.4.x embeds closed-over arrays in compiled programs. A mixed scene bank
    can exceed several GiB, so reset/rollout entry points receive its three field
    arrays explicitly. During Python tracing, the leaf environment temporarily
    reads those argument tracers; attributes are always restored afterwards.
    Compiled executions operate on arguments and never mutate Python objects.

    This object must only be used by the owning learner's tracing thread. The
    arrays are neither batched over environments nor added to checkpoint state.
    Ordinary non-bank environments bind an empty tuple and retain their methods.
    """

    def __init__(self, environment):
        self.environment = getattr(environment, "unwrapped", environment)
        self.names = ("sdf", "bf", "gf") if hasattr(self.environment, "field_bank_manifest") else ()
        if hasattr(self.environment, "_room_arrays"):
            self.names += ("_room_arrays", "_room_scene_index")
        self.values = tuple(getattr(self.environment, name) for name in self.names)

    @contextmanager
    def bind(self, values):
        if len(values) != len(self.names):
            raise ValueError("Dynamic scene-field arguments differ from the environment field tuple")
        previous = tuple(getattr(self.environment, name) for name in self.names)
        try:
            for name, value in zip(self.names, values):
                setattr(self.environment, name, value)
            yield
        finally:
            for name, value in zip(self.names, previous):
                setattr(self.environment, name, value)
