# MindIE-SD Backend 验证指南

本文档说明如何在 Docker 容器的 NPU 环境上验证 MindIE-SD 后端集成代码。

## 前置条件

### 1. 硬件要求

- 昇腾 NPU 设备（如 Atlas 800I A2）
- 至少 64GB 显存（推荐）
- 支持的操作系统：Linux（Ubuntu 20.04/22.04）

### 2. 软件要求

- Docker >= 20.10
- Docker Compose（可选）
- Python >= 3.10
- PyTorch >= 2.1.0

## 验证步骤

### 步骤 1: 准备 MindIE-SD 安装包

在宿主机上准备 MindIE-SD 安装包：

#### 方式 1: 源码编译安装

```bash
# 克隆 MindIE-SD 仓库
git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD

# 编译并安装
python setup.py bdist_wheel

# 安装生成的 wheel 包
cd dist
pip install mindiesd-*.whl
```

#### 方式 2: 直接 pip 安装（适用于 torch 2.6）

```bash
pip install --trusted-host ascend.devcloud.huaweicloud.com \
    -i https://ascend.devcloud.huaweicloud.com/pypi/simple/ mindiesd
```

**注意**：建议使用源码编译方式，以确保与 vllm-omni 的兼容性。

### 步骤 2: 构建 Docker 镜像

#### 2.1 修改 Dockerfile 以包含 MindIE-SD

创建或修改 `docker/Dockerfile.npu.mindie`：

```dockerfile
ARG VLLM_ASCEND_IMAGE=quay.io/ascend/vllm-ascend
ARG VLLM_ASCEND_TAG=v0.17.0rc1
FROM ${VLLM_ASCEND_IMAGE}:${VLLM_ASCEND_TAG}

ARG APP_DIR=/vllm-workspace/vllm-omni
WORKDIR ${APP_DIR}

# 复制 vllm-omni 代码
COPY . .

# 复制 MindIE-SD 安装包
COPY MindIE-SD/dist/mindiesd-*.whl /tmp/mindiesd.whl

# 设置环境变量
RUN export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/`uname -i`-linux/devlib && \
    # 安装 MindIE-SD
    pip install /tmp/mindiesd.whl && \
    # 安装 vllm-omni
    python3 -m pip install -v -e /vllm-workspace/vllm-omni/ --no-build-isolation

ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

ENTRYPOINT []
```

#### 2.2 构建镜像

```bash
cd /d/sourcecode/arch/vllm-omni-voxcpm

# 确保 MindIE-SD 的 wheel 包在 MindIE-SD/dist 目录下
# 然后构建镜像
docker build -f docker/Dockerfile.npu.mindie -t vllm-omni-mindie:latest .
```

### 步骤 3: 运行 Docker 容器

```bash
# 运行容器，挂载必要的目录
docker run -it \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /path/to/models:/models \
    -v /path/to/output:/output \
    --network host \
    vllm-omni-mindie:latest \
    /bin/bash
```

**参数说明：**
- `--device`: 挂载 NPU 设备
- `-v`: 挂载模型目录和输出目录
- `--network host`: 使用主机网络（可选）

### 步骤 4: 在容器中验证 MindIE-SD 安装

进入容器后，验证 MindIE-SD 是否正确安装：

```bash
# 激活环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 验证 MindIE-SD 安装
python -c "import mindiesd; print(mindiesd.__version__)"
python -c "from mindiesd.compilation import MindieSDBackend, CompilationConfig; print('MindIE-SD backend imported successfully')"

# 验证 torch_npu
python -c "import torch; import torch_npu; print(f'torch version: {torch.__version__}'); print(f'torch_npu available: {torch.npu.is_available()}')"
```

### 步骤 5: 验证集成代码

#### 5.1 测试基本导入

```bash
cd /vllm-workspace/vllm-omni

python -c "from vllm_omni.diffusion.mindie_sd_backend import MINDIE_SD_AVAILABLE, compile_with_mindie_sd, regionally_compile_with_mindie_sd, MindieSDCompiler; print('Import successful')"
```

#### 5.2 运行示例代码

创建一个简单的测试脚本 `test_mindie_sd.py`：

```python
#!/usr/bin/env python3

import torch
import torch.nn as nn
from vllm_omni.diffusion.mindie_sd_backend import (
    MINDIE_SD_AVAILABLE,
    compile_with_mindie_sd,
    MindieSDCompiler,
)

def test_basic_model():
    """测试基本模型编译"""
    if not MINDIE_SD_AVAILABLE:
        print("MindIE-SD is not available. Skipping test.")
        return False

    print("Testing basic model compilation...")

    # 创建一个简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(512, 512)
            self.norm = nn.LayerNorm(512)

        def forward(self, x):
            return self.norm(self.linear(x))

    model = SimpleModel()
    model.eval()

    # 使用 MindIE-SD 编译
    try:
        compiled_model = compile_with_mindie_sd(
            model,
            enable_freezing=True,
            enable_rms_norm=True,
            enable_rope=True,
            enable_adalayernorm=True,
            enable_fast_gelu=True,
            mode="reduce-overhead",
        )

        # 测试推理
        with torch.no_grad():
            x = torch.randn(1, 512)
            output = compiled_model(x)
            print(f"Output shape: {output.shape}")
            print("Basic model compilation test PASSED!")
            return True
    except Exception as e:
        print(f"Basic model compilation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compiler_class():
    """测试编译器类"""
    if not MINDIE_SD_AVAILABLE:
        print("MindIE-SD is not available. Skipping test.")
        return False

    print("\nTesting MindieSDCompiler class...")

    try:
        compiler = MindieSDCompiler(
            enable_freezing=True,
            enable_rms_norm=True,
            enable_rope=True,
            enable_adalayernorm=True,
            enable_fast_gelu=True,
        )

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(256, 256)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        model.eval()

        compiled_model = compiler.compile(
            model,
            mode="reduce-overhead",
        )

        with torch.no_grad():
            x = torch.randn(1, 256)
            output = compiled_model(x)
            print(f"Output shape: {output.shape}")
            print("MindieSDCompiler class test PASSED!")
            return True
    except Exception as e:
        print(f"MindieSDCompiler class test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("MindIE-SD Backend Integration Test")
    print("=" * 60)

    results = []

    results.append(("Basic Model Compilation", test_basic_model()))
    results.append(("Compiler Class", test_compiler_class()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result for _, result in results)
    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

运行测试：

```bash
python test_mindie_sd.py
```

#### 5.3 运行示例文件

```bash
# 运行示例代码（需要实际模型）
python vllm_omni/diffusion/mindie_sd_examples.py
```

### 步骤 6: 与实际模型集成测试

如果需要测试与实际 diffusion 模型的集成：

```python
#!/usr/bin/env python3

import torch
from vllm_omni.diffusion.mindie_sd_backend import compile_with_mindie_sd
from vllm_omni.diffusion.models.flux import FluxTransformer

# 加载模型
model = FluxTransformer.from_pretrained("/path/to/model")
model.eval()

# 使用 MindIE-SD 编译
compiled_model = compile_with_mindie_sd(
    model,
    enable_freezing=True,
    enable_rms_norm=True,
    enable_rope=True,
    enable_adalayernorm=True,
    enable_fast_gelu=True,
    mode="max-autotune",
    fullgraph=True,
)

# 测试推理
with torch.no_grad():
    # 根据模型要求准备输入
    inputs = ...
    outputs = compiled_model(**inputs)
    print(f"Output shape: {outputs.shape}")
```

## 常见问题排查

### 问题 1: MindIE-SD 导入失败

**症状**：`ImportError: No module named 'mindiesd'`

**解决方案**：
1. 确认 MindIE-SD 已正确安装
2. 检查 Python 路径是否包含 MindIE-SD 安装目录
3. 重新安装 MindIE-SD

### 问题 2: NPU 设备不可用

**症状**：`torch.npu.is_available()` 返回 False

**解决方案**：
1. 确认 NPU 驱动已正确安装
2. 检查 Docker 容器是否正确挂载了 NPU 设备
3. 确认环境变量已正确设置

### 问题 3: 编译错误

**症状**：编译过程中出现各种错误

**解决方案**：
1. 检查 PyTorch 版本是否兼容
2. 尝试禁用某些融合模式
3. 查看详细日志输出
4. 确认模型是否支持 torch.compile

### 问题 4: 内存不足

**症状**：`RuntimeError: CUDA out of memory` 或类似错误

**解决方案**：
1. 减小 batch size
2. 使用区域编译而非完整编译
3. 禁用图冻结
4. 使用更小的模型进行测试

## 性能优化建议

1. **使用区域编译**：对于大型模型，区域编译通常更高效
2. **启用图冻结**：`enable_freezing=True` 可以显著提高性能
3. **选择合适的融合模式**：根据模型特性选择合适的融合模式
4. **使用 max-autotune 模式**：`mode="max-autotune"` 可以获得最佳性能
5. **预热模型**：在正式推理前进行几次预热推理

## 验证清单

- [ ] MindIE-SD 已正确安装
- [ ] NPU 设备可用
- [ ] torch_npu 可正常工作
- [ ] 集成代码可以正常导入
- [ ] 基本模型编译测试通过
- [ ] 编译器类测试通过
- [ ] 实际模型集成测试通过（可选）

## 下一步

验证通过后，可以：

1. 在实际项目中使用 MindIE-SD 后端
2. 根据具体需求调整编译参数
3. 进行性能测试和优化
4. 集成到 vllm-omni 的 DiffusionEngine 中

## 参考资料

- [MindIE-SD 官方文档](https://gitcode.com/Ascend/MindIE-SD)
- [vllm-omni 文档](https://github.com/vllm-project/vllm-omni)
- [昇腾 NPU 文档](https://www.hiascend.com/document)

## 许可证

SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
