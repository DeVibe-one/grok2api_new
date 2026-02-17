#!/bin/bash

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "[错误] 请先运行 ./install.sh"
    exit 1
fi

source venv/bin/activate

echo "========================================"
echo "  Grok2API"
echo "  http://localhost:8000"
echo "  管理后台: http://localhost:8000/admin"
echo "========================================"
echo ""

python main.py
