# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from MindIE_SD_master.mindiesd.compilation import MindieSDBackend, CompilationConfig
    MINDIE_SD_AVAILABLE = True
except ImportError:
    MINDIE_SD_AVAILABLE = False
    logger.warning("MindIE-SD is not available. Please install MindIE-SD to use this backend.")


def compile_with_mindie_sd(
    model: nn.Module,
    enable_freezing: bool = True,
    enable_rms_norm: bool = True,
    enable_rope: bool = True,
    enable_adalayernorm: bool = True,
    enable_fast_gelu: bool = True,
    graph_log_url: str | None = None,
    *compile_args: Any,
    **compile_kwargs: Any
) -> nn.Module:
    """
    Compile a PyTorch model using MindIE-SD backend.

    Args:
        model: The PyTorch model instance to compile
        enable_freezing: Enable graph freezing optimization
        enable_rms_norm: Enable RMSNorm fusion pattern
        enable_rope: Enable RoPE fusion pattern
        enable_adalayernorm: Enable AdaLayerNorm fusion pattern
        enable_fast_gelu: Enable FastGELU fusion pattern
        graph_log_url: URL for graph logging (optional)
        *compile_args: Additional positional arguments for torch.compile
        **compile_kwargs: Additional keyword arguments for torch.compile

    Returns:
        The compiled model
    """
    if not MINDIE_SD_AVAILABLE:
        raise ImportError(
            "MindIE-SD is not available. Please install MindIE-SD to use this backend."
        )

    logger.info("Compiling model with MindIE-SD backend...")

    backend = MindieSDBackend()

    CompilationConfig.enable_freezing = enable_freezing
    CompilationConfig.graph_log_url = graph_log_url
    CompilationConfig.fusion_patterns.enable_rms_norm = enable_rms_norm
    CompilationConfig.fusion_patterns.enable_rope = enable_rope
    CompilationConfig.fusion_patterns.enable_adalayernorm = enable_adalayernorm
    CompilationConfig.fusion_patterns.enable_fast_gelu = enable_fast_gelu

    compiled_model = torch.compile(
        model,
        backend=backend,
        *compile_args,
        **compile_kwargs
    )

    logger.info("Model compilation with MindIE-SD backend completed.")
    return compiled_model


def regionally_compile_with_mindie_sd(
    model: nn.Module,
    enable_freezing: bool = True,
    enable_rms_norm: bool = True,
    enable_rope: bool = True,
    enable_adalayernorm: bool = True,
    enable_fast_gelu: bool = True,
    graph_log_url: str | None = None,
    *compile_args: Any,
    **compile_kwargs: Any
) -> nn.Module:
    """
    Apply regional compilation to a PyTorch model using MindIE-SD backend.

    Args:
        model: The PyTorch model instance to compile
        enable_freezing: Enable graph freezing optimization
        enable_rms_norm: Enable RMSNorm fusion pattern
        enable_rope: Enable RoPE fusion pattern
        enable_adalayernorm: Enable AdaLayerNorm fusion pattern
        enable_fast_gelu: Enable FastGELU fusion pattern
        graph_log_url: URL for graph logging (optional)
        *compile_args: Additional positional arguments for torch.compile
        **compile_kwargs: Additional keyword arguments for torch.compile

    Returns:
        The same model instance (modified in-place)
    """
    if not MINDIE_SD_AVAILABLE:
        raise ImportError(
            "MindIE-SD is not available. Please install MindIE-SD to use this backend."
        )

    logger.info("Applying regional compilation with MindIE-SD backend...")

    backend = MindieSDBackend()

    CompilationConfig.enable_freezing = enable_freezing
    CompilationConfig.graph_log_url = graph_log_url
    CompilationConfig.fusion_patterns.enable_rms_norm = enable_rms_norm
    CompilationConfig.fusion_patterns.enable_rope = enable_rope
    CompilationConfig.fusion_patterns.enable_adalayernorm = enable_adalayernorm
    CompilationConfig.fusion_patterns.enable_fast_gelu = enable_fast_gelu

    repeated_blocks = getattr(model, "_repeated_blocks", None)

    if not repeated_blocks:
        logger.warning(
            "Regional compilation skipped because the model does not define `_repeated_blocks`."
        )
        return model

    has_compiled_region = False
    for submod in model.modules():
        if submod.__class__.__name__ in repeated_blocks:
            compiled_submod = torch.compile(
                submod,
                backend=backend,
                *compile_args,
                **compile_kwargs
            )
            submod._compiled_forward = compiled_submod.forward
            submod.forward = compiled_submod.forward
            has_compiled_region = True

    if not has_compiled_region:
        logger.warning(
            f"Regional compilation skipped because {repeated_blocks} classes are not found in the model."
        )
    else:
        logger.info("Regional compilation with MindIE-SD backend completed.")

    return model


class MindieSDCompiler:
    """
    A compiler wrapper for MindIE-SD backend that provides a convenient interface
    for compiling diffusion models in vllm-omni.
    """

    def __init__(
        self,
        enable_freezing: bool = True,
        enable_rms_norm: bool = True,
        enable_rope: bool = True,
        enable_adalayernorm: bool = True,
        enable_fast_gelu: bool = True,
        graph_log_url: str | None = None
    ):
        """
        Initialize MindIE-SD compiler.

        Args:
            enable_freezing: Enable graph freezing optimization
            enable_rms_norm: Enable RMSNorm fusion pattern
            enable_rope: Enable RoPE fusion pattern
            enable_adalayernorm: Enable AdaLayerNorm fusion pattern
            enable_fast_gelu: Enable FastGELU fusion pattern
            graph_log_url: URL for graph logging (optional)
        """
        if not MINDIE_SD_AVAILABLE:
            raise ImportError(
                "MindIE-SD is not available. Please install MindIE-SD to use this backend."
            )

        self.enable_freezing = enable_freezing
        self.enable_rms_norm = enable_rms_norm
        self.enable_rope = enable_rope
        self.enable_adalayernorm = enable_adalayernorm
        self.enable_fast_gelu = enable_fast_gelu
        self.graph_log_url = graph_log_url

        self._configure_backend()

    def _configure_backend(self):
        """Configure MindIE-SD compilation settings."""
        CompilationConfig.enable_freezing = self.enable_freezing
        CompilationConfig.graph_log_url = self.graph_log_url
        CompilationConfig.fusion_patterns.enable_rms_norm = self.enable_rms_norm
        CompilationConfig.fusion_patterns.enable_rope = self.enable_rope
        CompilationConfig.fusion_patterns.enable_adalayernorm = self.enable_adalayernorm
        CompilationConfig.fusion_patterns.enable_fast_gelu = self.enable_fast_gelu

    def compile(
        self,
        model: nn.Module,
        *compile_args: Any,
        **compile_kwargs: Any
    ) -> nn.Module:
        """
        Compile a PyTorch model.

        Args:
            model: The PyTorch model instance to compile
            *compile_args: Additional positional arguments for torch.compile
            **compile_kwargs: Additional keyword arguments for torch.compile

        Returns:
            The compiled model
        """
        logger.info("Compiling model with MindIE-SD backend...")
        backend = MindieSDBackend()
        compiled_model = torch.compile(
            model,
            backend=backend,
            *compile_args,
            **compile_kwargs
        )
        logger.info("Model compilation with MindIE-SD backend completed.")
        return compiled_model

    def compile_regionally(
        self,
        model: nn.Module,
        *compile_args: Any,
        **compile_kwargs: Any
    ) -> nn.Module:
        """
        Apply regional compilation to a PyTorch model.

        Args:
            model: The PyTorch model instance to compile
            *compile_args: Additional positional arguments for torch.compile
            **compile_kwargs: Additional keyword arguments for torch.compile

        Returns:
            The same model instance (modified in-place)
        """
        logger.info("Applying regional compilation with MindIE-SD backend...")
        backend = MindieSDBackend()

        repeated_blocks = getattr(model, "_repeated_blocks", None)

        if not repeated_blocks:
            logger.warning(
                "Regional compilation skipped because the model does not define `_repeated_blocks`."
            )
            return model

        has_compiled_region = False
        for submod in model.modules():
            if submod.__class__.__name__ in repeated_blocks:
                compiled_submod = torch.compile(
                    submod,
                    backend=backend,
                    *compile_args,
                    **compile_kwargs
                )
                submod._compiled_forward = compiled_submod.forward
                submod.forward = compiled_submod.forward
                has_compiled_region = True

        if not has_compiled_region:
            logger.warning(
                f"Regional compilation skipped because {repeated_blocks} classes are not found in the model."
            )
        else:
            logger.info("Regional compilation with MindIE-SD backend completed.")

        return model


__all__ = [
    "MINDIE_SD_AVAILABLE",
    "compile_with_mindie_sd",
    "regionally_compile_with_mindie_sd",
    "MindieSDCompiler",
]
