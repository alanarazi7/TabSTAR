#!/bin/bash

# Simple script to test if 'pip install tabstar' works on Python 3.9
# Use this as a regression test after making code changes

echo "🐍 Testing: pip install tabstar on Python 3.9"
echo "============================================="

# Check if python3.9 is available
if ! command -v python3.9 &> /dev/null; then
    echo "❌ python3.9 not found"
    echo "Please install Python 3.9 to run this test"
    exit 1
fi

echo "✅ python3.9 found: $(python3.9 --version)"
echo ""

# Run the test
echo "🧪 Running pip installation test..."
python3.9 -m pytest test/test_python39_pip_install.py -v -s

echo ""
if [ $? -eq 0 ]; then
    echo "🎉 SUCCESS: pip install tabstar works on Python 3.9"
    echo "Your code changes are compatible with Python 3.9!"
else
    echo "❌ FAILURE: pip install tabstar broken on Python 3.9"
    echo "Your code changes may have broken Python 3.9 compatibility"
    exit 1
fi