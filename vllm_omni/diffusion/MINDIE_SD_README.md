# MindIE-SD Backend Integration for vllm-omni

这个模块提供了 MindIE-SD 编译后端与 vllm-omni diffusion 模型的集成。

## 功能特性

- **完整模型编译**: 使用 MindIE-SD 后端编译整个 diffusion 模型
- **区域编译**: 仅编译模型中的特定区域（如 transformer blocks）
- **自定义融合模式**: 支持配置不同的算子融合模式
- **图冻结优化**: 支持图冻结以提高性能
- **易于使用的 API**: 提供简单的函数和类接口

## 安装要求

确保已安装 MindIE-SD：

```bash
pip install MindIE-SD
```

## 快速开始

### 1. 完整模型编译

```python
from vllm_omni.diffusion.mindie_sd_backend import compile_with_mindie_sd
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
```

### 2. 区域编译

```python
from vllm_omni.diffusion.mindie_sd_backend import regionally_compile_with_mindie_sd
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
```

### 3. 使用编译器类

```python
from vllm_omni.diffusion.mindie_sd_backend import MindieSDCompiler
from vllm_omni.diffusion.models.flux import FluxTransformer

model = FluxTransformer(...)

compiler = MindieSDCompiler(
    enable_freezing=True,
    enable_rms_norm=True,
    enable_rope=True,
    enable_adalayernorm=True,
    enable_fast_gelu=True
)

compiled_model = compiler.compile(
    model,
    mode="max-autotune",
    fullgraph=True
)
```

## API 参考

### 函数接口

#### `compile_with_mindie_sd`

使用 MindIE-SD 后端编译整个模型。

**参数:**
- `model` (nn.Module): 要编译的 PyTorch 模型
- `enable_freezing` (bool): 启用图冻结优化，默认 True
- `enable_rms_norm` (bool): 启用 RMSNorm 融合模式，默认 True
- `enable_rope` (bool): 启用 RoPE 融合模式，默认 True
- `enable_adalayernorm` (bool): 启用 AdaLayerNorm 融合模式，默认 True
- `enable_fast_gelu` (bool): 启用 FastGELU 融合模式，默认 True
- `graph_log_url` (str | None): 图日志 URL，可选
- `*compile_args`: 传递给 torch.compile 的额外位置参数
- `**compile_kwargs`: 传递给 torch.compile 的额外关键字参数

**返回:**
- 编译后的模型 (nn.Module)

#### `regionally_compile_with_mindie_sd`

使用 MindIE-SD 后端对模型进行区域编译。

**参数:**
- 与 `compile_with_mindie_sd` 相同

**返回:**
- 编译后的模型 (nn.Module)，原地修改

### 类接口

#### `MindieSDCompiler`

MindIE-SD 编译器包装类，提供更灵活的编译控制。

**初始化参数:**
- `enable_freezing` (bool): 启用图冻结优化，默认 True
- `enable_rms_norm` (bool): 启用 RMSNorm 融合模式，默认 True
- `enable_rope` (bool): 启用 RoPE 融合模式，默认 True
- `enable_adalayernorm` (bool): 启用 AdaLayerNorm 融合模式，默认 True
- `enable_fast_gelu` (bool): 启用 FastGELU 融合模式，默认 True
- `graph_log_url` (str | None): 图日志 URL，可选

**方法:**

##### `compile(model, *compile_args, **compile_kwargs)`

编译整个模型。

##### `compile_regionally(model, *compile_args, **compile_kwargs)`

对模型进行区域编译。

## 融合模式说明

MindIE-SD 支持以下算子融合模式：

- **RMSNorm**: 根均方归一化融合
- **RoPE**: 旋转位置编码融合
- **AdaLayerNorm**: 自适应层归一化融合
- **FastGELU**: 快速 GELU 激活函数融合

可以根据模型特性选择性地启用或禁用这些融合模式。

## 性能优化建议

1. **使用区域编译**: 对于大型模型，区域编译通常比完整编译更高效
2. **启用图冻结**: `enable_freezing=True` 可以显著提高推理性能
3. **选择合适的融合模式**: 根据模型结构选择合适的融合模式
4. **使用 max-autotune 模式**: `mode="max-autotune"` 可以获得最佳性能

## 与 DiffusionEngine 集成

```python
from vllm_omni.diffusion.mindie_sd_backend import compile_with_mindie_sd
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.models.flux import FluxTransformer

model = FluxTransformer(...)

compiled_model = compile_with_mindie_sd(
    model,
    enable_freezing=True,
    mode="max-autotune",
    fullgraph=True
)

engine = DiffusionEngine(
    model=compiled_model,
    ...
)
```

## 故障排除

### 导入错误

如果遇到 "MindIE-SD is not available" 错误，请确保：

1. 已正确安装 MindIE-SD
2. MindIE-SD 在 Python 路径中可访问
3. MindIE-SD 版本与 PyTorch 版本兼容

### 编译错误

如果编译过程中遇到错误：

1. 检查模型是否支持 torch.compile
2. 尝试禁用某些融合模式
3. 查看日志输出以获取更多信息

## 示例

更多使用示例请参考 `mindie_sd_examples.py` 文件。

## 许可证

SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
