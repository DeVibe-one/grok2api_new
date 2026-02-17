#!/bin/bash

echo "========================================"
echo "  Grok2API 安装"
echo "========================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

echo "[1/2] 创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv || { echo "[错误] 创建虚拟环境失败"; exit 1; }
else
    echo "       已存在，跳过"
fi

source venv/bin/activate

echo "[2/2] 安装依赖..."
pip install -r requirements.txt -q

echo ""
echo "========================================"
echo "  安装完成，运行 ./start.sh 启动服务"
echo "========================================"
