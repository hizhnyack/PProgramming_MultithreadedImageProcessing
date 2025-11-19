#!/bin/bash

echo "=================================="
echo "  CUDA Image Processing Tests"
echo "=================================="
echo ""

cd build

# Счетчик пройденных тестов
PASSED=0
FAILED=0

# Тест 1: Grayscale
echo "1. Testing Grayscale Filter..."
if ./test_grayscale > /dev/null 2>&1; then
    echo "   ✓ PASSED"
    ((PASSED++))
else
    echo "   ✗ FAILED"
    ((FAILED++))
fi

# Тест 2: Rotation
echo "2. Testing Rotation Filter..."
if ./test_rotation > /dev/null 2>&1; then
    echo "   ✓ PASSED"
    ((PASSED++))
else
    echo "   ✗ FAILED"
    ((FAILED++))
fi

# Тест 3: Blur
echo "3. Testing Blur Filter..."
if ./test_blur > /dev/null 2>&1; then
    echo "   ✓ PASSED"
    ((PASSED++))
else
    echo "   ✗ FAILED"
    ((FAILED++))
fi

echo ""
echo "=================================="
echo "  Results: $PASSED passed, $FAILED failed"
echo "=================================="

if [ $FAILED -eq 0 ]; then
    echo "  🎉 All tests PASSED!"
    exit 0
else
    echo "  ❌ Some tests FAILED"
    exit 1
fi

