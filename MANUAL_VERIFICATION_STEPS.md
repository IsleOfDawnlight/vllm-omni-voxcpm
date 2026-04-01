# MindIE-SD Backend 手动验证步骤

本文档提供手动验证 MindIE-SD 后端集成的详细步骤，用于排查自动化脚本中的问题。

## 🔍 问题排查指南

如果 `verify_mindie_sd.sh` 脚本中途退出且没有报错，可能的原因：

1. **`set -e` 导致静默失败** - 任何命令返回非零退出码都会导致脚本立即退出
2. **环境变量文件不存在** - `/usr/local/Ascend/ascend-toolkit/set_env.sh` 等文件可能不存在
3. **Python 导入失败** - 某些 Python 包可能未正确安装
4. **文件路径问题** - 测试文件路径可能不正确

## 📝 手动验证步骤

### 步骤 1: 检查环境

```bash
# 检查是否在容器中
ls -la /.dockerenv

# 检查工作目录
pwd
ls -la
```

### 步骤 2: 检查环境变量文件

```bash
# 检查昇腾环境变量文件是否存在
ls -la /usr/local/Ascend/ascend-toolkit/set_env.sh
ls -la /usr/local/Ascend/nnal/atb/set_env.sh

# 如果文件存在，尝试加载
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 检查环境变量
echo $LD_LIBRARY_PATH
echo $ASCEND_HOME
```

### 步骤 3: 检查 Python

```bash
# 检查 Python 版本
python --version

# 检查 Python 路径
which python
python -c "import sys; print(sys.path)"
```

### 步骤 4: 检查 PyTorch

```bash
# 检查 PyTorch 版本
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# 检查 torch_npu
python -c "import torch_npu; print(f'torch_npu available: {torch.npu.is_available()}'); print(f'NPU device count: {torch.npu.device_count() if torch.npu.is_available() else 0}')"

# 如果失败，查看详细错误
python -c "import torch_npu" 2>&1
```

### 步骤 5: 检查 MindIE-SD

```bash
# 检查 MindIE-SD 是否安装
pip list | grep mindiesd

# 尝试导入 MindIE-SD
python -c "import mindiesd; print(f'MindIE-SD version: {mindiesd.__version__}')"

# 如果失败，查看详细错误
python -c "import mindiesd" 2>&1

# 检查 MindIE-SD 后端
python -c "from mindiesd.compilation import MindieSDBackend, CompilationConfig; print('MindIE-SD backend imported successfully')"

# 如果失败，查看详细错误
python -c "from mindiesd.compilation import MindieSDBackend, CompilationConfig" 2>&1
```

### 步骤 6: 检查 vllm-omni

```bash
# 检查 vllm 是否安装
pip list | grep vllm

# 尝试导入 vllm
python -c "import vllm; print(f'vllm version: {vllm.__version__}')"

# 如果失败，查看详细错误
python -c "import vllm" 2>&1
```

### 步骤 7: 检查集成代码

```bash
# 检查集成代码文件是否存在
ls -la vllm_omni/diffusion/mindie_sd_backend.py
ls -la vllm_omni/diffusion/test_mindie_sd.py

# 尝试导入集成代码
python -c "from vllm_omni.diffusion.mindie_sd_backend import MINDIE_SD_AVAILABLE, compile_with_mindie_sd, regionally_compile_with_mindie_sd, MindieSDCompiler; print('Integration code imported successfully')"

# 如果失败，查看详细错误
python -c "from vllm_omni.diffusion.mindie_sd_backend import MINDIE_SD_AVAILABLE, compile_with_mindie_sd, regionally_compile_with_mindie_sd, MindieSDCompiler" 2>&1
```

### 步骤 8: 运行测试

```bash
# 直接运行测试脚本
python vllm_omni/diffusion/test_mindie_sd.py

# 如果失败，查看详细错误
python vllm_omni/diffusion/test_mindie_sd.py 2>&1
```

## 🛠️ 常见问题解决方案

### 问题 1: 环境变量文件不存在

**症状**：
```
source: /usr/local/Ascend/ascend-toolkit/set_env.sh: No such file or directory
```

**解决方案**：
```bash
# 查找环境变量文件
find /usr/local/Ascend -name "set_env.sh" 2>/dev/null

# 如果找到，使用实际路径
source /path/to/found/set_env.sh
```

### 问题 2: MindIE-SD 未安装

**症状**：
```
ModuleNotFoundError: No module named 'mindiesd'
```

**解决方案**：
```bash
# 检查 MindIE-SD wheel 包
ls -la /tmp/mindiesd*.whl

# 重新安装 MindIE-SD
pip install /tmp/mindiesd*.whl

# 或者从源码安装
cd /path/to/MindIE-SD
python setup.py bdist_wheel
pip install dist/mindiesd-*.whl
```

### 问题 3: vllm-omni 未安装

**症状**：
```
ModuleNotFoundError: No module named 'vllm'
```

**解决方案**：
```bash
# 重新安装 vllm-omni
cd /vllm-workspace/vllm-omni
python3 -m pip install -e . --no-build-isolation
```

### 问题 4: torch_npu 不可用

**症状**：
```
torch_npu available: False
```

**解决方案**：
```bash
# 检查 NPU 设备
ls -la /dev/davinci*

# 检查 NPU 驱动
npu-smi info

# 重新安装 torch_npu
pip install torch-npu
```

### 问题 5: 集成代码导入失败

**症状**：
```
ModuleNotFoundError: No module named 'vllm_omni.diffusion.mindie_sd_backend'
```

**解决方案**：
```bash
# 检查 Python 路径
python -c "import sys; print('\n'.join(sys.path))"

# 确保在正确的目录
cd /vllm-workspace/vllm-omni

# 重新安装 vllm-omni
python3 -m pip install -e . --no-build-isolation
```

## 🔧 使用改进的验证脚本

我创建了一个改进的验证脚本 `verify_mindie_sd_debug.sh`，它具有以下改进：

1. **移除了 `set -e`** - 不会因为单个命令失败而退出
2. **详细的错误处理** - 每个步骤都有错误捕获和报告
3. **错误计数器** - 统计总错误数
4. **彩色输出** - 更清晰的成功/失败标识
5. **调试信息** - 显示每个命令的输出

使用方法：

```bash
# 赋予执行权限
chmod +x verify_mindie_sd_debug.sh

# 运行改进的脚本
./verify_mindie_sd_debug.sh
```

## 📊 逐步调试流程

如果仍然遇到问题，按照以下流程逐步调试：

1. **首先运行改进的脚本**
   ```bash
   ./verify_mindie_sd_debug.sh
   ```

2. **记录失败的步骤**
   - 注意哪个步骤失败
   - 查看错误消息

3. **针对失败的步骤进行手动验证**
   - 使用上面的手动验证步骤
   - 逐个命令执行

4. **查看详细错误输出**
   - 使用 `2>&1` 捕获所有输出
   - 查看完整的错误堆栈

5. **根据错误类型采取相应措施**
   - 参考上面的常见问题解决方案
   - 查阅相关文档

## 📞 获取帮助

如果以上步骤都无法解决问题，请收集以下信息：

```bash
# 收集系统信息
uname -a
docker --version

# 收集 Python 环境信息
python --version
pip list

# 收集 NPU 信息
npu-smi info

# 收集错误日志
./verify_mindie_sd_debug.sh > verification.log 2>&1
```

然后提供这些信息以获取进一步的帮助。

## ✅ 验证成功标志

当所有步骤都成功时，你应该看到：

```
========================================
All tests passed successfully!
========================================

You can now use MindIE-SD backend in your vllm-omni projects.

Example usage:
  from vllm_omni.diffusion.mindie_sd_backend import compile_with_mindie_sd
  compiled_model = compile_with_mindie_sd(model, mode='max-autotune')
```

## 📚 相关文档

- [快速开始指南](./QUICKSTART_MINDIE_SD.md)
- [详细验证指南](./vllm_omni/diffusion/MINDIE_SD_VERIFICATION_GUIDE.md)
- [MindIE-SD 官方文档](https://gitcode.com/Ascend/MindIE-SD)
