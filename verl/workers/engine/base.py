# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The abstract base class defining the interface for model training engines.
"""

import os
from abc import abstractmethod
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Generator, Optional

import torch
from tensordict import TensorDict

from verl.utils.device import get_device_name, get_vendor
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids

import logging

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class BaseEngine:
    """
    Abstract base class defining the interface for model training engines. Interface is subject to
    change before release.

    Engine implementations must subclass BaseEngine and provide concrete behavior for all methods.
    """

    def initialize(self):
        """
        Instantiate or load the model, optimizer, and learning rate scheduler.

        Should prepare all components necessary for training or evaluation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_param_offload_enabled(self) -> bool:
        """Whether parameter offloading is enabled."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_optimizer_offload_enabled(self) -> bool:
        """Whether optimizer offloading is enabled."""
        raise NotImplementedError

    def train_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into training mode.

        Usage:
            with engine.train_mode():
                # runs in training mode
        """
        raise NotImplementedError

    def eval_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into evaluation mode.

        Usage:
            with engine.eval_mode():
                # runs in evaluation mode
        """
        raise NotImplementedError

    def optimizer_zero_grad(self):
        """
        Zero the gradients of the optimizer.
        """
        raise NotImplementedError

    def optimizer_step(self):
        """
        Perform an optimization step using the optimizer.
        """
        raise NotImplementedError

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        raise NotImplementedError

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        """
        Perform a forward pass and optionally a backward pass on a batch of data.

        Args:
            data: The input data for the forward pass, typically containing tensors and metadata.
            loss_function: The loss function to optimize. See `verl.workers.roles.utils.losses` for examples.
            forward_only: If True, perform only the forward pass. If False, perform forward and backward pass.

        Returns:
            Any: The output of the forward pass, which can be used for loss computation or other purposes.
        """
        # ================================ NOTE：这是整条训练的「核心黑盒」 ================================
        # 抽象桩，真正实现在具体引擎子类里（我们跑的是 fsdp/transformer_impl.py 的 FSDPEngine.forward_backward_batch
        #   + FSDPEngineWithLMHead.forward_step）。下面把「输入 / 输出契约」讲清楚，方便写自定义 loss 时知道该传什么。
        #
        # 【一句话】SFT、PPO、OPD(蒸馏)、DPO 的「前向 + 反向」流程完全一样：
        #   micro-batch 切分 → 模型前向得到 model_output → 调 loss_function(model_output, data) 得标量 loss →
        #   forward_only=False 时 loss.backward()。换算法只是换 loss_function，引擎这层不用动。
        #
        # 【loss_function 的 API】——所有算法都遵守这一个签名：
        #   loss_function(model_output: dict, data: TensorDict, dp_group) -> (loss: Tensor[标量], metrics: dict)
        #   注意它只有两个数据入参：model_output 和 data。这两者的分工，恰好就是「梯度」的分界线：
        #
        # 【model_output = 唯一的梯度通道】
        #     反向传播 loss.backward() 只能穿过「带 grad_fn 的可导张量」回到模型参数。而：
        #       · model_output 是 self.module(...) 前向刚产出的张量，带 grad_fn ——loss 对它求导，梯度就回到模型参数；
        #       · data 里的一切都是常量：input_ids/mask 不可导；old_log_probs/advantages/ref_log_prob 是上层提前算好、
        #         detach 过的系数。loss 里它们只当权重/掩码/目标，不回传梯度。
        #     所以自定义 loss 的本质就是 loss = f( model_output[可导] , data[常量] )：
        #     想让某个量参与梯度，它必须来自 model_output（模型前向的产物）；只当条件/系数的量，放 data 就行。
        #
        #   model_output 的字段（forward_step→prepare_model_outputs 从 logits 组装，都带 grad_fn）：
        #     - "log_probs" ：必有，response token 的 log 概率——绝大多数 loss 的求导对象（PPO/SFT/OPD/DPO 都用它）；
        #     - "entropy"   ：可选（熵正则时才有）；   "values"：critic/value 模型才有。
        #   返回：
        #     - loss    ：单个标量 tensor（可导），直接拿去 backward（一般已按全局 token 归一化，见 /batch_num_tokens*dp_size）；
        #     - metrics ：dict，值可为 Metric(带 SUM/MEAN 聚合语义) 或原始标量，用于日志聚合（pg_loss、kl_loss、grad_norm 等）。
        #
        # 【引擎硬依赖的 data 输入。不是直接用于回传梯度的部分，但是需要用于计算 loss专门用于定制优化算法】
        #   data 里绝大多数 key 引擎不看（留给 loss 自取），但这几个是引擎前向/归一化自己要用的，务必保证有：
        #     - input_ids / attention_mask / position_ids —— forward_step 里 prepare_model_inputs 组装后喂给 self.module 前向；
        #     - loss_mask —— 引擎入口用它算 batch_num_tokens = all_reduce(SUM, loss_mask.sum())，作全局 token 归一化分母。
        #   引擎还会自动往 data 塞回：batch_num_tokens、dp_size、sp_size，供 loss 做全局归一化。
        #
        # 【其余 data 随意——整个 data 原样传给 loss，loss 自己用 data["xxx"] 挑】
        #   除上面引擎必需项外，其它 key 引擎一概不看，只是原样切进 micro-batch、原样交给 loss_function。
        #   要什么额外常量输入，上层构 batch 时塞进 data 即可，引擎不用改：
        #     - 逐样本/逐 token 张量（随 batch 切）——放 data 的 tensor 字段：PPO 的 old_log_probs/advantages、
        #       OPD 的老师端 teacher_logprob/topk、DPO 的 ref_log_prob/is_chosen，都走这条路，没有特殊通道；
        #     - 全局标量/超参（不随样本变）——挂 config 上（partial(loss_fn, config=...) 读 config.beta），
        #       或 tu.assign_non_tensor(data, k=v) 再 tu.get_non_tensor_data 取。
        #   （DPO 额外注意：chosen/rejected 别被 micro-batch 切散；ref_log_prob 提前跑一趟 infer_batch 算好再塞 data。）
        #
        # 【loss_function 返回】(model_output, data) 进去，吐出两样：
        #     - loss    ：单个标量 tensor（可导），引擎直接 loss.backward()（一般已按全局 token 归一化，见 /batch_num_tokens*dp_size）；
        #     - metrics ：dict，值可为 Metric(带 SUM/MEAN 聚合语义) 或原始标量，用于日志聚合（pg_loss、kl_loss 等）。
        #
        # 【本函数返回】把 loss_function 的产物 postprocess 成 list[TensorDict]/dict，含三部分：
        #     - "model_output" ：模型前向产物（log_probs 等），infer_batch(如 compute_log_prob) 用它；
        #     - "loss"         ：标量 float（已 detach，仅供记录，不再回传）；
        #     - "metrics"      ：dict，各项训练/评估指标，train_batch 主要用它（含 grad_norm 等）。
        #   一句话：train_batch 只用 loss/metrics；infer_batch 用 model_output。
        # =============================================================================================
        raise NotImplementedError

    def train_batch(self, data: TensorDict, loss_function: Callable) -> Any:
        """
        Perform a training step on a batch of data.

        Args:
            data: The input data for training, typically containing tensors and metadata.
            loss_function: A function that computes the loss and metrics given a batch and predictions.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the aggregated training metrics for the batch.
        """
        maybe_fix_3d_position_ids(data)

        # NOTE：train = zero_grad → forward_backward_batch(forward_only=False，含 loss.backward()) → optimizer_step。
        #   跟 SFT 训练步骤完全一致，区别只在 loss_function 是谁。这里 log 一下这一步的输入/输出契约，方便调试自定义 loss。
        logger.debug(
            "[engine.train_batch] loss_fn=%s | input data keys=%s | loss_fn 需要的典型条目：log_probs(来自前向)、"
            "loss_mask / response_mask / old_log_probs / advantages（视算法而定）、dp_size / batch_num_tokens（归一化）",
            getattr(loss_function, "func", loss_function).__name__ if loss_function is not None else None,
            list(data.keys()),
        )

        self.optimizer_zero_grad()
        outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
        grad_norm = self.optimizer_step()
        if self.is_mp_src_rank_with_outputs():
            assert "grad_norm" not in outputs["metrics"]
            outputs["metrics"]["grad_norm"] = grad_norm
            # NOTE：输出契约——outputs 含 "loss"(标量)、"metrics"(dict，此处补进 grad_norm)，训练只用这两样。
            logger.debug("[engine.train_batch] output metrics keys=%s", list(outputs["metrics"].keys()))
        return outputs

    def infer_batch(self, data: TensorDict, loss_function: Optional[Callable] = None) -> Any:
        """
        Perform inference on a batch of data.

        Args:
            data: The input data for inference, typically containing tensors and metadata.

        Returns:
            Any: The output of the inference, which can be used for predictions or other purposes.
        """
        # see comments from train_batch
        maybe_fix_3d_position_ids(data)

        # NOTE：infer = 和 train 落到同一个 forward_backward_batch，只是 forward_only=True 且包在 no_grad 里：
        #   只前向、不反向、不更新。loss_function 可为 None（如只算 log_probs/values）；SFT/eval 需要顺带算 loss 时才传。
        with torch.no_grad():
            outputs = self.forward_backward_batch(data, loss_function, forward_only=True)
        return outputs

    def get_per_tensor_param(self) -> tuple[Generator[tuple[str, torch.Tensor], None, None], Optional[dict]]:
        """
        Get a generator that yields per-tensor parameters and optional peft config.

        Returns:
            Generator[tuple[str, torch.Tensor]]: A generator that yields tuples of parameter names and tensors.
            Optional[dict]: Optional peft config.
        """
        raise NotImplementedError

    def get_data_parallel_size(self):
        raise NotImplementedError

    def get_data_parallel_rank(self):
        raise NotImplementedError

    def get_data_parallel_group(self):
        raise NotImplementedError

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
            grad: If True, move the gradient buffer.
        """
        if grad:
            assert model, "Gradient buffers must be moved to device along with model parameters"

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
            **kwargs: Arbitrary keyword arguments.
        """
        raise NotImplementedError

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True, **kwargs
    ) -> None:
        """
        Load model, optimizer, and scheduler states from a checkpoint.

        Args:
            local_path: Local filesystem path of the checkpoint.
            hdfs_path: Optional HDFS path where checkpoint is stored.
            del_local_after_load: Whether to delete local copy after loading.
            **kwargs: Arbitrary keyword arguments.
        """
        raise NotImplementedError

    def is_mp_src_rank_with_outputs(self):
        """
        Whether the current rank is the first rank in model parallel group that contains model outputs
        """
        raise NotImplementedError

    def disable_adapter(self) -> ContextManager:
        """
        Disable all adapters temporarily under the context in the model for LoRA
        """
        return nullcontext()


class BaseEngineCtx:
    def __init__(self, engine: BaseEngine, mode, **kwargs):
        """Base Engine context that handles load and offload

        Args:
            engine:
            **kwargs:
        """
        self.engine = engine
        self.mode = mode
        assert self.mode in ("train", "eval")
        self.disable_auto_offload = kwargs.pop("disable_auto_offload", False)

    def _context_switch(self, device):
        if self.disable_auto_offload:
            return
        if device != "cpu":
            if not self.engine.is_param_offload_enabled and not self.engine.is_optimizer_offload_enabled:
                return
        if self.mode == "eval":
            self.engine.to(device=device, model=self.engine.is_param_offload_enabled, optimizer=False, grad=False)
        elif self.mode == "train":
            self.engine.to(
                device=device,
                model=self.engine.is_param_offload_enabled,
                optimizer=self.engine.is_optimizer_offload_enabled,
                grad=self.engine.is_param_offload_enabled,
            )

    def __enter__(self):
        self.engine.mode = self.mode
        self._context_switch(get_device_name())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context_switch("cpu")
        self.engine.mode = None


class EngineRegistry:
    """
    A registry for managing and instantiating different types of training engines.

    This class uses a dictionary to store engine classes, mapping a string key to each class.
    It provides a decorator `register` to add new engines to the registry and a `new` method
    to create an instance of a registered engine.
    """

    _engines = {}

    @classmethod
    def register(
        cls,
        model_type: str,
        backend: list[str] | str,
        device: list[str] | str = "cuda",
        vendor: list[str] | str | None = None,
    ):
        """
        A class method decorator that registers an engine class with a given key.

        This allows for dynamic instantiation of engine classes by their registered key.

        Args:
            model_type (str): The type of the model
            backend (list[str] | str): The backend to use for the model type
            device (list[str] | str): The device type (e.g., "cuda", "npu", "cpu") this engine supports,
                default is "cuda"
            vendor (list[str] | str | None): The hardware vendor (e.g., "nvidia", "metax") this engine
                supports. If None, the engine is registered as the default for the device type.

        Returns:
            A decorator function that takes an engine class and registers it.
        """

        def decorator(engine_class):
            assert issubclass(engine_class, BaseEngine)
            if model_type not in cls._engines:
                cls._engines[model_type] = {}

            backends = backend if isinstance(backend, list) else [backend]
            devices = device if isinstance(device, list) else [device]
            vendors = vendor if isinstance(vendor, list) else ([vendor] if vendor else [None])
            for current_backend in backends:
                for current_device in devices:
                    if current_backend not in cls._engines[model_type]:
                        cls._engines[model_type][current_backend] = {}
                    for current_vendor in vendors:
                        key = (current_device, current_vendor) if current_vendor else current_device
                        assert key not in cls._engines[model_type][current_backend], (
                            f"The key(device-vendor: {key}) has been already registed!"
                        )
                        cls._engines[model_type][current_backend][key] = engine_class

            return engine_class

        return decorator

    @classmethod
    def get_engine_cls(cls, model_type: str, backend: str):
        assert model_type in cls._engines, f"Unknown model_type: {model_type}"
        assert backend in cls._engines[model_type], f"Unknown backend: {backend}"
        device = get_device_name()
        vendor = get_vendor()
        # Allow environment variables to override detected device and vendor for engine selection, if set
        if os.getenv("VERL_ENGINE_DEVICE"):
            device = os.getenv("VERL_ENGINE_DEVICE")
        if os.getenv("VERL_ENGINE_VENDOR"):
            vendor = os.getenv("VERL_ENGINE_VENDOR")
        registry = cls._engines[model_type][backend]

        # Try vendor-specific lookup: (device, vendor)
        vendor_key = (device, vendor)
        if vendor_key in registry:
            return registry[vendor_key]

        # Fallback to device-only key (registered without vendor)
        if device in registry:
            return registry[device]

        # For cuda-compatible vendors without a specific registration, try nvidia
        if device == "cuda" and vendor != "nvidia":
            nvidia_key = (device, "nvidia")
            if nvidia_key in registry:
                return registry[nvidia_key]

        raise ValueError(
            f"No engine registered for device={device!r}, vendor={vendor!r}, "
            f"model_type={model_type!r}, backend={backend!r}"
        )

    @classmethod
    def new(cls, model_type, backend, *args, **kwargs):
        """
        Function to create a new training engine instance based on the provided config.
        Args:
            key: A configuration object containing the engine key and other settings.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            engine: An instance of the training engine corresponding to the config.
        Raises:
            NotImplementedError: If the engine key in the config does not match any known engines.
        """
        engine_cls = cls.get_engine_cls(model_type, backend)
        return engine_cls(*args, **kwargs)
