#!/bin/bash

# 安装中文字体脚本（适用于WSL/Ubuntu）

echo "正在安装中文字体支持..."

# 更新包列表
sudo apt update

# 安装中文字体
sudo apt install -y fonts-noto-cjk fonts-wqy-microhei fonts-wqy-zenhei

# 清除字体缓存
sudo fc-cache -fv

# 检查已安装的字体
echo "已安装的中文字体："
fc-list :lang=zh

echo "字体安装完成！"