#!/bin/bash

# Script to install development tools for FEM Numeric Library

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Installing development tools for FEM Numeric Library..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        echo -e "${GREEN}Detected Debian/Ubuntu${NC}"
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            lcov \
            gcovr \
            valgrind \
            clang-tools \
            cppcheck \
            doxygen \
            graphviz

    elif command -v dnf &> /dev/null; then
        # Fedora/RHEL/CentOS
        echo -e "${GREEN}Detected Fedora/RHEL/CentOS${NC}"
        sudo dnf install -y \
            gcc-c++ \
            cmake \
            lcov \
            gcovr \
            valgrind \
            clang-tools-extra \
            cppcheck \
            doxygen \
            graphviz

    elif command -v pacman &> /dev/null; then
        # Arch Linux
        echo -e "${GREEN}Detected Arch Linux${NC}"
        sudo pacman -S --needed \
            base-devel \
            cmake \
            lcov \
            gcovr \
            valgrind \
            clang \
            cppcheck \
            doxygen \
            graphviz

    elif command -v apk &> /dev/null; then
        # Alpine Linux
        echo -e "${GREEN}Detected Alpine Linux${NC}"
        sudo apk add \
            build-base \
            cmake \
            lcov \
            gcovr \
            valgrind \
            clang-extra-tools \
            cppcheck \
            doxygen \
            graphviz
    else
        echo -e "${RED}Unknown Linux distribution${NC}"
        echo "Please install: cmake, lcov, gcovr, valgrind manually"
        exit 1
    fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo -e "${GREEN}Detected macOS${NC}"

    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo -e "${YELLOW}Homebrew not found. Installing...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    brew install \
        cmake \
        lcov \
        gcovr \
        cppcheck \
        doxygen \
        graphviz

    echo -e "${YELLOW}Note: Valgrind is not well supported on recent macOS versions${NC}"
    echo "Consider using AddressSanitizer instead (built into clang)"

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows with MSYS2/Cygwin
    echo -e "${GREEN}Detected Windows (MSYS2/Cygwin)${NC}"
    pacman -S --needed \
        mingw-w64-x86_64-gcc \
        mingw-w64-x86_64-cmake \
        mingw-w64-x86_64-lcov \
        mingw-w64-x86_64-doxygen

    echo -e "${YELLOW}Note: Some tools may not be available on Windows${NC}"
    echo "Consider using WSL2 for full Linux toolchain support"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

# Install Python-based tools (cross-platform)
echo -e "${GREEN}Installing Python-based tools...${NC}"
if command -v pip3 &> /dev/null; then
    pip3 install --user gcovr
elif command -v pip &> /dev/null; then
    pip install --user gcovr
else
    echo -e "${YELLOW}pip not found. Skipping Python tools${NC}"
fi

echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "To verify installation:"
echo "  cmake --version"
echo "  lcov --version"
echo "  gcovr --version"
echo "  valgrind --version"