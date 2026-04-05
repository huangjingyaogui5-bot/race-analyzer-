#!/bin/bash
# Vercel ビルドスクリプト
# 環境変数 RAILWAY_API_URL を index.html の __API_URL__ プレースホルダーに注入する

set -e

mkdir -p dist

# RAILWAY_API_URL が設定されていればプレースホルダーを置換
if [ -n "$RAILWAY_API_URL" ]; then
  echo "API URL: $RAILWAY_API_URL"
  sed "s|__API_URL__|$RAILWAY_API_URL|g" index.html > dist/index.html
else
  echo "警告: RAILWAY_API_URL が設定されていません。localhost を使用します。"
  cp index.html dist/index.html
fi

echo "ビルド完了: dist/index.html"
