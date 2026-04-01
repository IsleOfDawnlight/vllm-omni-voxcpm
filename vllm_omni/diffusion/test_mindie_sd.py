#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from vllm_omni.diffusion.mindie_sd_backend import (
        MINDIE_SD_AVAILABLE,
        compile_with_mindie_sd,
        regionally_compile_with_mindie_sd,
        MindieSDCompiler,
    )
except ImportError as e:
    print(f"Error importing mindie_sd_backend: {e}")
    print("Please ensure vllm-omni is installed correctly.")
    sys.exit(1)


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing"""

    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x):
        B, L, D = x.shape

        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhqk,bhvd->bhqd", attn, v)
        out = out.reshape(B, L, D)
        out = self.proj(out)

        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleDiffusionModel(nn.Module):
    """Simple diffusion model for testing"""

    def __init__(self, hidden_size=512, num_layers=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            SimpleTransformerBlock(hidden_size) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

        self._repeated_blocks = ["SimpleTransformerBlock"]

    def forward(self, x, timesteps):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def test_mindie_sd_availability():
    """Test if MindIE-SD is available"""
    print("=" * 60)
    print("Test 1: MindIE-SD Availability")
    print("=" * 60)

    if MINDIE_SD_AVAILABLE:
        print("✓ MindIE-SD is available")

        try:
            from mindiesd.compilation import MindieSDBackend, CompilationConfig
            print("✓ MindieSDBackend imported successfully")
            print("✓ CompilationConfig imported successfully")
            return True
        except ImportError as e:
            print(f"✗ Failed to import MindIE-SD components: {e}")
            return False
    else:
        print("✗ MindIE-SD is not available")
        print("  Please install MindIE-SD first:")
        print("  pip install mindiesd")
        return False


def test_basic_compilation():
    """Test basic model compilation"""
    print("\n" + "=" * 60)
    print("Test 2: Basic Model Compilation")
    print("=" * 60)

    if not MINDIE_SD_AVAILABLE:
        print("✗ Skipped: MindIE-SD is not available")
        return False

    try:
        model = SimpleTransformerBlock()
        model.eval()

        print("Compiling model with MindIE-SD backend...")
        compiled_model = compile_with_mindie_sd(
            model,
            enable_freezing=True,
            enable_rms_norm=True,
            enable_rope=True,
            enable_adalayernorm=True,
            enable_fast_gelu=True,
            mode="reduce-overhead",
        )

        print("✓ Model compiled successfully")

        with torch.no_grad():
            x = torch.randn(1, 32, 512)
            output = compiled_model(x)
            print(f"✓ Inference successful, output shape: {output.shape}")

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_regional_compilation():
    """Test regional compilation"""
    print("\n" + "=" * 60)
    print("Test 3: Regional Compilation")
    print("=" * 60)

    if not MINDIE_SD_AVAILABLE:
        print("✗ Skipped: MindIE-SD is not available")
        return False

    try:
        model = SimpleDiffusionModel()
        model.eval()

        print("Compiling model regionally with MindIE-SD backend...")
        compiled_model = regionally_compile_with_mindie_sd(
            model,
            enable_freezing=True,
            enable_rms_norm=True,
            enable_rope=True,
            enable_adalayernorm=True,
            enable_fast_gelu=True,
            mode="reduce-overhead",
        )

        print("✓ Model compiled regionally successfully")

        with torch.no_grad():
            x = torch.randn(1, 32, 512)
            timesteps = torch.tensor([0.5])
            output = compiled_model(x, timesteps)
            print(f"✓ Inference successful, output shape: {output.shape}")

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compiler_class():
    """Test MindieSDCompiler class"""
    print("\n" + "=" * 60)
    print("Test 4: MindieSDCompiler Class")
    print("=" * 60)

    if not MINDIE_SD_AVAILABLE:
        print("✗ Skipped: MindIE-SD is not available")
        return False

    try:
        compiler = MindieSDCompiler(
            enable_freezing=True,
            enable_rms_norm=True,
            enable_rope=True,
            enable_adalayernorm=True,
            enable_fast_gelu=True,
        )
        print("✓ MindieSDCompiler initialized successfully")

        model = SimpleTransformerBlock()
        model.eval()

        print("Compiling model using MindieSDCompiler...")
        compiled_model = compiler.compile(
            model,
            mode="reduce-overhead",
        )

        print("✓ Model compiled successfully using MindieSDCompiler")

        with torch.no_grad():
            x = torch.randn(1, 32, 512)
            output = compiled_model(x)
            print(f"✓ Inference successful, output shape: {output.shape}")

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_fusion_patterns():
    """Test custom fusion patterns"""
    print("\n" + "=" * 60)
    print("Test 5: Custom Fusion Patterns")
    print("=" * 60)

    if not MINDIE_SD_AVAILABLE:
        print("✗ Skipped: MindIE-SD is not available")
        return False

    try:
        model = SimpleTransformerBlock()
        model.eval()

        print("Compiling model with custom fusion patterns...")
        compiled_model = compile_with_mindie_sd(
            model,
            enable_freezing=True,
            enable_rms_norm=True,
            enable_rope=False,
            enable_adalayernorm=True,
            enable_fast_gelu=False,
            mode="reduce-overhead",
        )

        print("✓ Model compiled with custom fusion patterns")

        with torch.no_grad():
            x = torch.randn(1, 32, 512)
            output = compiled_model(x)
            print(f"✓ Inference successful, output shape: {output.shape}")

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_npu_availability():
    """Test NPU availability"""
    print("\n" + "=" * 60)
    print("Test 6: NPU Availability")
    print("=" * 60)

    try:
        import torch_npu

        if torch.npu.is_available():
            device_count = torch.npu.device_count()
            print(f"✓ NPU is available")
            print(f"  Number of NPU devices: {device_count}")
            for i in range(device_count):
                device_name = torch.npu.get_device_name(i)
                print(f"  Device {i}: {device_name}")
            return True
        else:
            print("✗ NPU is not available")
            print("  This is expected if running without NPU hardware")
            return False
    except ImportError:
        print("✗ torch_npu is not installed")
        print("  This is expected if running on non-NPU hardware")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("MindIE-SD Backend Integration Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("MindIE-SD Availability", test_mindie_sd_availability),
        ("Basic Compilation", test_basic_compilation),
        ("Regional Compilation", test_regional_compilation),
        ("Compiler Class", test_compiler_class),
        ("Custom Fusion Patterns", test_custom_fusion_patterns),
        ("NPU Availability", test_npu_availability),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for test_name, result in results:
        if result is True:
            status = "PASSED ✓"
            passed += 1
        elif result is False:
            status = "FAILED ✗"
            failed += 1
        else:
            status = "SKIPPED -"
            skipped += 1
        print(f"{test_name:.<40} {status}")

    print("=" * 60)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print("=" * 60)

    if failed == 0:
        print("\n✓ All critical tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed!")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
