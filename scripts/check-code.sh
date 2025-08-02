#!/bin/bash

# Code quality check script for F1 What-If Simulator API

set -e

echo "🔍 Running code quality checks..."

echo "📝 Running pre-commit hooks..."
pre-commit run --all-files

echo "🧪 Running tests..."
python3 -m pytest tests/ -v

echo "✅ All checks passed!"
