#!/bin/bash

# Script to clean in-source build artifacts from the core library

echo "Cleaning in-source build artifacts..."

# Remove CMake cache and generated files
rm -f CMakeCache.txt
rm -f cmake_install.cmake
rm -f Makefile
rm -f compile_commands.json
rm -f CoreConfig.cmake
rm -f CTestTestfile.cmake

# Remove CMake directories
rm -rf CMakeFiles/
rm -rf _deps/
rm -rf Testing/

# Remove build outputs
rm -f libcore.a
rm -rf lib/
rm -rf bin/

# Remove test-generated files in tests/
find tests/ -name "*.cmake" -type f -delete 2>/dev/null || true
find tests/ -name "CTestTestfile.cmake" -type f -delete 2>/dev/null || true

echo "Source directory cleaned!"
echo ""
echo "To perform a clean out-of-source build:"
echo "  mkdir -p build"
echo "  cd build"
echo "  cmake .."
echo "  make -j$(nproc)"