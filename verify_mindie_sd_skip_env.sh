#!/bin/bash

# MindIE-SD Backend 验证脚本（临时修复版）
# 跳过环境变量加载，直接进行验证

set -e

echo "========================================"
echo "MindIE-SD Backend Verification Script (Skip Env)"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在容器中
if [ ! -f /.dockerenv ]; then
    echo -e "${YELLOW}Warning: This script is designed to run inside a Docker container${NC}"
    echo "Please run this script inside a Docker container"
    echo ""
fi

# 跳过环境变量加载（临时方案）
echo "Step 1: Skipping environment variables activation (temporary fix)..."
echo -e "${YELLOW}Note: Environment variables may not be set. If you encounter issues, please set them manually.${NC}"
echo ""

# 检查 Python 版本
echo "Step 2: Checking Python version..."
python --version
echo ""

# 检查 torch 和 torch_npu
echo "Step 3: Checking PyTorch and torch_npu..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch_npu; print(f'torch_npu available: {torch.npu.is_available()}'); print(f'NPU device count: {torch.npu.device_count() if torch.npu.is_available() else 0}')"
echo ""
  
# 检查 MindIE-SD 安装
echo "Step 4: Checking MindIE-SD installation..."
python -c "import mindiesd; print(f'MindIE-SD version: {mindiesd.__version__}')" 2>/dev/null || {
    echo -e "${RED}✗ MindIE-SD is not installed${NC}"
    echo "Please install MindIE-SD first:"
    echo "  pip install mindiesd"
    exit 1
}
echo -e "${GREEN}✓ MindIE-SD is installed${NC}"
echo ""

# 检查 MindIE-SD 后端
echo "Step 5: Checking MindIE-SD backend..."
python -c "from mindiesd.compilation import MindieSDBackend, CompilationConfig; print('MindIE-SD backend imported successfully')" || {
    echo -e "${RED}✗ Failed to import MindIE-SD backend${NC}"
    exit 1
}
echo -e "${GREEN}✓ MindIE-SD backend is available${NC}"
echo ""

# 检查 vllm-omni 安装
echo "Step 6: Checking vllm-omni installation..."
python -c "import vllm; print(f'vllm version: {vllm.__version__}')" || {
    echo -e "${RED}✗ vllm is not installed${NC}"
    exit 1
}
echo -e "${GREEN}✓ vllm is installed${NC}"
echo ""

# 检查集成代码
echo "Step 7: Checking MindIE-SD backend integration..."
python -c "from vllm_omni.diffusion.mindie_sd_backend import MINDIE_SD_AVAILABLE, compile_with_mindie_sd, regionally_compile_with_mindie_sd, MindieSDCompiler; print('Integration code imported successfully')" || {
    echo -e "${RED}✗ Failed to import integration code${NC}"
    exit 1
}
echo -e "${GREEN}✓ Integration code is available${NC}"
echo ""

# 运行测试
echo "Step 8: Running integration tests..."
echo ""
if [ -f "vllm_omni/diffusion/test_mindie_sd.py" ]; then
    python vllm_omni/diffusion/test_mindie_sd.py
    TEST_RESULT=$?
    echo ""

    if [ $TEST_RESULT -eq 0 ]; then
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
        exit 1
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
