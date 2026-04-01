#!/bin/bash

# MindIE-SD Backend 验证脚本（改进版）
# 用于在 Docker 容器中快速验证 MindIE-SD 后端集成
# 添加了详细的错误处理和调试信息

# 不使用 set -e，改为手动错误处理
# set -e  # 已禁用，改为手动处理错误

echo "========================================"
echo "MindIE-SD Backend Verification Script"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 错误计数器
ERROR_COUNT=0

# 错误处理函数
handle_error() {
    local step_name="$1"
    local error_msg="$2"
    echo -e "${RED}✗ Error in $step_name${NC}"
    echo -e "${RED}  $error_msg${NC}"
    ERROR_COUNT=$((ERROR_COUNT + 1))
}

# 成功处理函数
handle_success() {
    local msg="$1"
    echo -e "${GREEN}✓ $msg${NC}"
}

# 检查是否在容器中
echo "Step 0: Checking environment..."
if [ ! -f /.dockerenv ]; then
    echo -e "${YELLOW}Warning: This script is designed to run inside a Docker container${NC}"
    echo "Please run this script inside a Docker container"
    echo ""
fi
handle_success "Environment check completed"
echo ""

# 激活环境变量
echo "Step 1: Activating environment variables..."
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>&1 || {
        handle_error "Environment setup" "Failed to source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    }
else
    handle_error "Environment setup" "File not found: /usr/local/Ascend/ascend-toolkit/set_env.sh"
fi

if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh 2>&1 || {
        handle_error "Environment setup" "Failed to source /usr/local/Ascend/nnal/atb/set_env.sh"
    }
else
    handle_error "Environment setup" "File not found: /usr/local/Ascend/nnal/atb/set_env.sh"
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/$(uname -i)-linux/devlib
handle_success "Environment variables activated"
echo ""

# 检查 Python 版本
echo "Step 2: Checking Python version..."
if command -v python &> /dev/null; then
    python --version
    handle_success "Python found"
else
    handle_error "Python check" "Python command not found"
fi
echo ""

# 检查 torch 和 torch_npu
echo "Step 3: Checking PyTorch and torch_npu..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>&1 || {
    handle_error "PyTorch check" "Failed to import torch"
}

python -c "import torch_npu; print(f'torch_npu available: {torch.npu.is_available()}'); print(f'NPU device count: {torch.npu.device_count() if torch.npu.is_available() else 0}')" 2>&1 || {
    handle_error "torch_npu check" "Failed to import torch_npu"
}
handle_success "PyTorch and torch_npu check completed"
echo ""

# 检查 MindIE-SD 安装
echo "Step 4: Checking MindIE-SD installation..."
python -c "import mindiesd; print(f'MindIE-SD version: {mindiesd.__version__}')" 2>&1
if [ $? -eq 0 ]; then
    handle_success "MindIE-SD is installed"
else
    handle_error "MindIE-SD check" "MindIE-SD is not installed or import failed"
    echo "  Please install MindIE-SD first:"
    echo "    pip install mindiesd"
fi
echo ""

# 检查 MindIE-SD 后端
echo "Step 5: Checking MindIE-SD backend..."
python -c "from mindiesd.compilation import MindieSDBackend, CompilationConfig; print('MindIE-SD backend imported successfully')" 2>&1
if [ $? -eq 0 ]; then
    handle_success "MindIE-SD backend is available"
else
    handle_error "MindIE-SD backend check" "Failed to import MindIE-SD backend"
fi
echo ""

# 检查 vllm-omni 安装
echo "Step 6: Checking vllm-omni installation..."
python -c "import vllm; print(f'vllm version: {vllm.__version__}')" 2>&1
if [ $? -eq 0 ]; then
    handle_success "vllm is installed"
else
    handle_error "vllm check" "vllm is not installed or import failed"
fi
echo ""

# 检查集成代码
echo "Step 7: Checking MindIE-SD backend integration..."
python -c "from vllm_omni.diffusion.mindie_sd_backend import MINDIE_SD_AVAILABLE, compile_with_mindie_sd, regionally_compile_with_mindie_sd, MindieSDCompiler; print('Integration code imported successfully')" 2>&1
if [ $? -eq 0 ]; then
    handle_success "Integration code is available"
else
    handle_error "Integration code check" "Failed to import integration code"
fi
echo ""

# 运行测试
echo "Step 8: Running integration tests..."
echo ""

if [ -f "vllm_omni/diffusion/test_mindie_sd.py" ]; then
    echo -e "${BLUE}Running test suite...${NC}"
    python vllm_omni/diffusion/test_mindie_sd.py 2>&1
    TEST_RESULT=$?
    echo ""

    if [ $TEST_RESULT -eq 0 ]; then
        handle_success "All tests passed"
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}All tests passed successfully!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "You can now use MindIE-SD backend in your vllm-omni projects."
        echo ""
        echo "Example usage:"
        echo "  from vllm_omni.diffusion.mindie_sd_backend import compile_with_mindie_sd"
        echo "  compiled_model = compile_with_mindie_sd(model, mode='max-autotune')"
        echo ""
    else
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}Some tests failed!${NC}"
        echo -e "${RED}========================================${NC}"
        echo ""
        echo "Please check the error messages above for details."
        ERROR_COUNT=$((ERROR_COUNT + 1))
    fi
else
    echo -e "${YELLOW}Warning: Test file not found at vllm_omni/diffusion/test_mindie_sd.py${NC}"
    echo "Skipping automated tests..."
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Setup check completed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "You can manually run tests:"
    echo "  python vllm_omni/diffusion/test_mindie_sd.py"
    echo ""
fi

# 总结
echo ""
echo "========================================"
echo "Verification Summary"
echo "========================================"
echo "Total errors: $ERROR_COUNT"
echo "========================================"

if [ $ERROR_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ $ERROR_COUNT error(s) occurred${NC}"
    echo ""
    echo "Please review the error messages above and fix the issues."
    exit 1
fi
