# MindIE-SD Backend 快速开始指南

本指南提供在 Docker 容器的 NPU 环境上快速验证 MindIE-SD 后端集成的步骤。

## 📋 前提条件

- ✅ 昇腾 NPU 硬件（如 Atlas 800I A2）
- ✅ Docker >= 20.10
- ✅ MindIE-SD 源码或安装包
- ✅ vllm-omni 源码

## 🚀 快速验证步骤

### 1. 准备 MindIE-SD 安装包

在宿主机上执行：

```bash
# 进入 MindIE-SD 目录
cd /path/to/MindIE-SD

# 编译并安装
python setup.py bdist_wheel

# 检查生成的 wheel 包
ls -lh dist/mindiesd-*.whl
```

### 2. 准备文件结构

确保以下文件结构：

```
d:\sourcecode\arch\
├── MindIE-SD-master\
│   └── dist\
│       └── mindiesd-*.whl  ← MindIE-SD 安装包
└── vllm-omni-voxcpm\
    ├── docker\
    │   └── Dockerfile.npu.mindie  ← Dockerfile
    ├── vllm_omni\
    │   └── diffusion\
    │       ├── mindie_sd_backend.py  ← 集成代码
    │       ├── test_mindie_sd.py  ← 测试脚本
    │       └── mindie_sd_examples.py  ← 示例代码
    └── verify_mindie_sd.sh  ← 验证脚本
```

### 3. 构建 Docker 镜像

```bash
# 进入 vllm-omni 目录
cd d:\sourcecode\arch\vllm-omni-voxcpm

# 构建 Docker 镜像
docker build -f docker/Dockerfile.npu.mindie -t vllm-omni-mindie:latest .
```

**注意**：确保 MindIE-SD 的 wheel 包在 `MindIE-SD-master/dist/` 目录下。

### 4. 运行 Docker 容器

```bash
# 运行容器（单卡）
docker run -it \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /path/to/models:/models \
    -v /path/to/output:/output \
    --network host \
    --name vllm-omni-mindie \
    vllm-omni-mindie:latest \
    /bin/bash

# 运行容器（多卡）
docker run -it \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /path/to/models:/models \
    -v /path/to/output:/output \
    --network host \
    --name vllm-omni-mindie \
    vllm-omni-mindie:latest \
    /bin/bash
```

### 5. 在容器中验证

进入容器后，运行验证脚本：

```bash
# 赋予执行权限
chmod +x verify_mindie_sd.sh

# 运行验证脚本
./verify_mindie_sd.sh
```

或者手动运行测试：

```bash
# 激活环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 运行测试
python vllm_omni/diffusion/test_mindie_sd.py
```

### 6. 验证结果

测试脚本会自动运行以下测试：

- ✅ MindIE-SD 可用性检查
- ✅ 基本模型编译
- ✅ 区域编译
- ✅ 编译器类测试
- ✅ 自定义融合模式
- ✅ NPU 可用性检查

所有测试通过后，你将看到：

```
========================================
All tests passed successfully!
========================================

You can now use MindIE-SD backend in your vllm-omni projects.

Example usage:
  from vllm_omni.diffusion.mindie_sd_backend import compile_with_mindie_sd
  compiled_model = compile_with_mindie_sd(model, mode='max-autotune')
```

## 💡 使用示例

### 基本使用

```python
from vllm_omni.diffusion.mindie_sd_backend import compile_with_mindie_sd
from vllm_omni.diffusion.models.flux import FluxTransformer

# 加载模型
model = FluxTransformer.from_pretrained("/models/flux-model")
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
    fullgraph=True
)

# 使用编译后的模型进行推理
with torch.no_grad():
    outputs = compiled_model(**inputs)
```

### 区域编译

```python
from vllm_omni.diffusion.mindie_sd_backend import regionally_compile_with_mindie_sd

# 仅编译模型的特定区域
compiled_model = regionally_compile_with_mindie_sd(
    model,
    enable_freezing=True,
    mode="max-autotune",
    fullgraph=True
)
```

### 使用编译器类

```python
from vllm_omni.diffusion.mindie_sd_backend import MindieSDCompiler

# 创建编译器实例
compiler = MindieSDCompiler(
    enable_freezing=True,
    enable_rms_norm=True,
    enable_rope=True,
    enable_adalayernorm=True,
    enable_fast_gelu=True
)

# 编译模型
compiled_model = compiler.compile(
    model,
    mode="max-autotune",
    fullgraph=True
)
```

## 🔧 常见问题

### 问题 1: MindIE-SD 导入失败

```bash
# 检查 MindIE-SD 是否安装
python -c "import mindiesd; print(mindiesd.__version__)"

# 如果未安装，重新安装
pip install /tmp/mindiesd.whl
```

### 问题 2: NPU 设备不可用

```bash
# 检查 NPU 设备
python -c "import torch_npu; print(torch.npu.is_available())"

# 检查设备权限
ls -l /dev/davinci*
```

### 问题 3: 编译错误

```bash
# 尝试使用更简单的编译模式
compiled_model = compile_with_mindie_sd(
    model,
    mode="reduce-overhead",  # 而不是 "max-autotune"
    fullgraph=False  # 禁用 fullgraph
)
```

## 📚 更多资源

- [详细验证指南](./MINDIE_SD_VERIFICATION_GUIDE.md)
- [API 文档](./MINDIE_SD_README.md)
- [示例代码](./mindie_sd_examples.py)
- [MindIE-SD 官方文档](https://gitcode.com/Ascend/MindIE-SD)

## 🎯 下一步

验证通过后，你可以：

1. 在实际项目中使用 MindIE-SD 后端
2. 根据具体需求调整编译参数
3. 进行性能测试和优化
4. 集成到 vllm-omni 的 DiffusionEngine 中

## 📝 验证清单

- [ ] MindIE-SD 已正确安装
- [ ] Docker 镜像构建成功
- [ ] Docker 容器可以正常运行
- [ ] NPU 设备在容器中可用
- [ ] 集成代码可以正常导入
- [ ] 所有测试通过
- [ ] 可以成功编译模型
- [ ] 可以进行推理

---

**祝使用愉快！** 🎉
