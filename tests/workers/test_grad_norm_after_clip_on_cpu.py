# [memdata patch 2026-11-26] The Megatron engine reports the gradient norm after clipping next to the
# (pre-clip) grad_norm, so logs show clipping took effect.

from types import SimpleNamespace
from unittest.mock import MagicMock

from tensordict import TensorDict

from verl.workers.engine.base import BaseEngine
from verl.workers.engine.megatron.transformer_impl import MegatronEngine


def test_megatron_optimizer_step_records_norm_after_clipping():
    """optimizer.step() returns the pre-clip norm; the norm of the (clipped) grads afterwards is kept too."""
    engine = SimpleNamespace(optimizer=MagicMock())
    engine.optimizer.step.return_value = (True, 34.3, None)  # pre-clip norm, as Megatron returns it
    engine.optimizer.get_grad_norm.return_value = 1.0  # grads were scaled by 1 / 34.3 inside step()

    assert MegatronEngine.optimizer_step(engine) == 34.3
    assert engine.grad_norm_after_clip == 1.0
    engine.optimizer.get_grad_norm.assert_called_once_with()


def test_train_batch_reports_both_norms():
    """BaseEngine.train_batch puts grad_norm and, when the engine recorded it, grad_norm_after_clip in the metrics."""

    class Engine(BaseEngine):
        def __init__(self, after):
            self.after = after

        def optimizer_zero_grad(self):
            pass

        def forward_backward_batch(self, data, loss_function, forward_only=False):
            return {"metrics": {}}

        def optimizer_step(self):
            if self.after is not None:
                self.grad_norm_after_clip = self.after
            return 6.1

        def is_mp_src_rank_with_outputs(self):
            return True

    data = TensorDict({}, batch_size=[])
    assert Engine(1.0).train_batch(data, loss_function=None)["metrics"] == {"grad_norm": 6.1, "grad_norm_after_clip": 1.0}
    assert Engine(None).train_batch(data, loss_function=None)["metrics"] == {"grad_norm": 6.1}
