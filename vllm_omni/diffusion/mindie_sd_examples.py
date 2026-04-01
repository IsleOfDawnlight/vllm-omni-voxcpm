# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example usage of MindIE-SD backend in vllm-omni diffusion models.

This module demonstrates how to integrate and use the MindIE-SD compilation
backend with vllm-omni diffusion models.
"""

import torch
import torch.nn as nn
from vllm_omni.diffusion.mindie_sd_backend import (
    MindieSDCompiler,
    compile_with_mindie_sd,
    regionally_compile_with_mindie_sd,
    MINDIE_SD_AVAILABLE,
)


def example_full_model_compilation():
    """
    Example: Compile an entire diffusion model with MindIE-SD backend.
    """
    if not MINDIE_SD_AVAILABLE:
        print("MindIE-SD is not available. Skipping example.")
        return

    print("Example 1: Full model compilation")
    print("-" * 50)

    from vllm_omni.diffusion.models.flux import FluxTransformer

    model = FluxTransformer(...)
    
    compiled_model = compile_with_mindie_sd(
        model,
        enable_freezing=True,
        enable_rms_norm=True,
        enable_rope=True,
        enable_adalayernorm=True,
        enable_fast_gelu=True,
        mode="max-autotune",
        fullgraph=True
    )

    print("Model compiled successfully!")
    print()


def example_regional_compilation():
    """
    Example: Compile specific regions of a diffusion model with MindIE-SD backend.
    """
    if not MINDIE_SD_AVAILABLE:
        print("MindIE-SD is not available. Skipping example.")
        return

    print("Example 2: Regional compilation")
    print("-" * 50)

    from vllm_omni.diffusion.models.flux import FluxTransformer

    model = FluxTransformer(...)
    
    compiled_model = regionally_compile_with_mindie_sd(
        model,
        enable_freezing=True,
        enable_rms_norm=True,
        enable_rope=True,
        enable_adalayernorm=True,
        enable_fast_gelu=True,
        mode="max-autotune",
        fullgraph=True
    )

    print("Model regionally compiled successfully!")
    print()


def example_compiler_class():
    """
    Example: Use MindieSDCompiler class for more control over compilation.
    """
    if not MINDIE_SD_AVAILABLE:
        print("MindIE-SD is not available. Skipping example.")
        return

    print("Example 3: Using MindieSDCompiler class")
    print("-" * 50)

    from vllm_omni.diffusion.models.flux import FluxTransformer

    model = FluxTransformer(...)

    compiler = MindieSDCompiler(
        enable_freezing=True,
        enable_rms_norm=True,
        enable_rope=True,
        enable_adalayernorm=True,
        enable_fast_gelu=True,
        graph_log_url=None
    )

    compiled_model = compiler.compile(
        model,
        mode="max-autotune",
        fullgraph=True
    )

    print("Model compiled using MindieSDCompiler class!")
    print()


def example_regional_compiler_class():
    """
    Example: Use MindieSDCompiler class for regional compilation.
    """
    if not MINDIE_SD_AVAILABLE:
        print("MindIE-SD is not available. Skipping example.")
        return

    print("Example 4: Regional compilation using MindieSDCompiler class")
    print("-" * 50)

    from vllm_omni.diffusion.models.flux import FluxTransformer

    model = FluxTransformer(...)

    compiler = MindieSDCompiler(
        enable_freezing=True,
        enable_rms_norm=True,
        enable_rope=True,
        enable_adalayernorm=True,
        enable_fast_gelu=True
    )

    compiled_model = compiler.compile_regionally(
        model,
        mode="max-autotune",
        fullgraph=True
    )

    print("Model regionally compiled using MindieSDCompiler class!")
    print()


def example_custom_fusion_patterns():
    """
    Example: Customize fusion patterns for specific use cases.
    """
    if not MINDIE_SD_AVAILABLE:
        print("MindIE-SD is not available. Skipping example.")
        return

    print("Example 5: Custom fusion patterns")
    print("-" * 50)

    from vllm_omni.diffusion.models.flux import FluxTransformer

    model = FluxTransformer(...)

    compiled_model = compile_with_mindie_sd(
        model,
        enable_freezing=True,
        enable_rms_norm=True,
        enable_rope=False,
        enable_adalayernorm=True,
        enable_fast_gelu=True,
        mode="max-autotune",
        fullgraph=True
    )

    print("Model compiled with custom fusion patterns!")
    print()


def example_with_diffusion_engine():
    """
    Example: Integrate MindIE-SD compilation with vllm-omni diffusion engine.
    """
    if not MINDIE_SD_AVAILABLE:
        print("MindIE-SD is not available. Skipping example.")
        return

    print("Example 6: Integration with diffusion engine")
    print("-" * 50)

    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.models.flux import FluxTransformer

    model = FluxTransformer(...)

    compiled_model = compile_with_mindie_sd(
        model,
        enable_freezing=True,
        enable_rms_norm=True,
        enable_rope=True,
        enable_adalayernorm=True,
        enable_fast_gelu=True,
        mode="max-autotune",
        fullgraph=True
    )

    engine = DiffusionEngine(
        model=compiled_model,
        ...
    )

    print("Diffusion engine created with MindIE-SD compiled model!")
    print()


def main():
    """
    Run all examples.
    """
    print("=" * 50)
    print("MindIE-SD Backend Examples for vllm-omni")
    print("=" * 50)
    print()

    example_full_model_compilation()
    example_regional_compilation()
    example_compiler_class()
    example_regional_compiler_class()
    example_custom_fusion_patterns()
    example_with_diffusion_engine()

    print("=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
