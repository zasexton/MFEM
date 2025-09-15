#!/bin/bash
# install-dev-tools.sh - Install development tools for FEM Core Library

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${GREEN}=== $1 ===${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ ${NC}$1"
}

print_warning() {
    echo -e "${YELLOW}⚠ ${NC}$1"
}

print_error() {
    echo -e "${RED}✗ ${NC}$1"
}

print_success() {
    echo -e "${GREEN}✓ ${NC}$1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect package manager
detect_package_manager() {
    if command_exists apt; then
        echo "apt"
    elif command_exists yum; then
        echo "yum"
    elif command_exists dnf; then
        echo "dnf"
    elif command_exists pacman; then
        echo "pacman"
    elif command_exists brew; then
        echo "brew"
    else
        echo "unknown"
    fi
}

# Install packages based on package manager
install_packages() {
    local pm=$1
    shift
    local packages=("$@")

    case $pm in
        apt)
            print_info "Installing packages with apt..."
            sudo apt update
            sudo apt install -y "${packages[@]}"
            ;;
        yum)
            print_info "Installing packages with yum..."
            sudo yum install -y "${packages[@]}"
            ;;
        dnf)
            print_info "Installing packages with dnf..."
            sudo dnf install -y "${packages[@]}"
            ;;
        pacman)
            print_info "Installing packages with pacman..."
            sudo pacman -S --noconfirm "${packages[@]}"
            ;;
        brew)
            print_info "Installing packages with brew..."
            brew install "${packages[@]}"
            ;;
        *)
            print_error "Unknown package manager. Please install manually:"
            for pkg in "${packages[@]}"; do
                echo "  - $pkg"
            done
            return 1
            ;;
    esac
}

print_header "FEM Core Library Development Tools Setup"

# Detect system
pm=$(detect_package_manager)
print_info "Detected package manager: $pm"

# Core development tools
print_header "Installing Core Development Tools"

case $pm in
    apt)
        packages=(
            "build-essential"
            "cmake"
            "git"
            "lcov"
            "gcovr"
            "valgrind"
            "clang"
            "clang-format"
            "clang-tidy"
            "cppcheck"
            "doxygen"
            "graphviz"
        )
        ;;
    yum|dnf)
        packages=(
            "gcc-c++"
            "cmake"
            "git"
            "lcov"
            "gcovr"
            "valgrind"
            "clang"
            "clang-tools-extra"
            "cppcheck"
            "doxygen"
            "graphviz"
        )
        ;;
    pacman)
        packages=(
            "base-devel"
            "cmake"
            "git"
            "lcov"
            "gcovr"
            "valgrind"
            "clang"
            "clang-tools-extra"
            "cppcheck"
            "doxygen"
            "graphviz"
        )
        ;;
    brew)
        packages=(
            "cmake"
            "git"
            "lcov"
            "gcovr"
            "valgrind"
            "llvm"
            "cppcheck"
            "doxygen"
            "graphviz"
        )
        ;;
    *)
        print_error "Please install the following tools manually:"
        echo "  - C++ compiler (GCC 11+ or Clang 12+)"
        echo "  - CMake 3.20+"
        echo "  - Git"
        echo "  - lcov (for coverage reports)"
        echo "  - gcovr (alternative coverage tool)"
        echo "  - Valgrind (for memory analysis)"
        echo "  - Clang tools (format, tidy)"
        echo "  - Cppcheck (static analysis)"
        echo "  - Doxygen (documentation)"
        echo "  - Graphviz (for diagrams)"
        exit 1
        ;;
esac

if ! install_packages "$pm" "${packages[@]}"; then
    print_error "Failed to install some packages"
    exit 1
fi

print_success "Core development tools installed"

# Check versions
print_header "Checking Tool Versions"

tools=(
    "gcc --version | head -1"
    "g++ --version | head -1"
    "cmake --version | head -1"
    "git --version"
    "lcov --version | head -1"
    "gcovr --version"
    "valgrind --version | head -1"
)

if command_exists clang; then
    tools+=("clang --version | head -1")
fi

if command_exists clang-format; then
    tools+=("clang-format --version")
fi

if command_exists clang-tidy; then
    tools+=("clang-tidy --version")
fi

if command_exists cppcheck; then
    tools+=("cppcheck --version")
fi

if command_exists doxygen; then
    tools+=("doxygen --version")
fi

for tool_cmd in "${tools[@]}"; do
    echo -n "  "
    eval "$tool_cmd" 2>/dev/null || echo "Not available: $tool_cmd"
done

# Create useful development aliases/scripts
print_header "Setting Up Development Environment"

# Create a development configuration script
cat > dev-setup.sh << 'EOF'
#!/bin/bash
# Development environment setup for FEM Core Library

# Build configurations
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Debug}
export CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER:-g++}
export CMAKE_C_COMPILER=${CMAKE_C_COMPILER:-gcc}

# Coverage configuration
export GCOV_TOOL=${GCOV_TOOL:-gcov}

# Add common aliases
alias core-build='cmake -B build -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE && make -C build -j$(nproc)'
alias core-test='cd build && ctest --output-on-failure'
alias core-coverage='cd build && ../scripts/coverage.sh'
alias core-detailed-coverage='cd build && ../scripts/detailed_line_coverage.sh'
alias core-clean='rm -rf build/ && mkdir build'

# Development shortcuts
alias core-format='find . -name "*.h" -o -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i'
alias core-lint='find . -name "*.cpp" | xargs clang-tidy'
alias core-check='cppcheck --enable=all --std=c++20 .'

echo "FEM Core development environment loaded!"
echo "Available commands:"
echo "  core-build          - Build the project"
echo "  core-test           - Run tests"
echo "  core-coverage       - Generate coverage reports"
echo "  core-detailed-coverage - Generate detailed line coverage"
echo "  core-clean          - Clean build directory"
echo "  core-format         - Format code with clang-format"
echo "  core-lint           - Run clang-tidy"
echo "  core-check          - Run cppcheck static analysis"
EOF

chmod +x dev-setup.sh
print_success "Created dev-setup.sh with useful aliases"

# Create .clang-format if it doesn't exist
if [ ! -f ../.clang-format ]; then
    print_info "Creating .clang-format configuration..."
    cat > ../.clang-format << 'EOF'
---
Language: Cpp
BasedOnStyle: Google
IndentWidth: 4
TabWidth: 4
UseTab: Never
ColumnLimit: 100
AccessModifierOffset: -2
AlignAfterOpenBracket: Align
AlignConsecutiveAssignments: false
AlignConsecutiveDeclarations: false
AlignOperands: true
AlignTrailingComments: true
AllowAllParametersOfDeclarationOnNextLine: false
AllowShortBlocksOnASingleLine: false
AllowShortCaseLabelsOnASingleLine: false
AllowShortFunctionsOnASingleLine: Empty
AllowShortIfStatementsOnASingleLine: false
AllowShortLoopsOnASingleLine: false
BinPackArguments: false
BinPackParameters: false
BreakBeforeBraces: Attach
BreakBeforeTernaryOperators: true
BreakConstructorInitializersBeforeComma: false
BreakStringLiterals: true
Cpp11BracedListStyle: true
DerivePointerAlignment: false
IndentCaseLabels: true
IndentWrappedFunctionNames: false
KeepEmptyLinesAtTheStartOfBlocks: false
MaxEmptyLinesToKeep: 1
NamespaceIndentation: None
PointerAlignment: Left
SpaceAfterCStyleCast: false
SpaceBeforeAssignmentOperators: true
SpaceBeforeParens: ControlStatements
SpaceInEmptyParentheses: false
SpacesInAngles: false
SpacesInContainerLiterals: false
SpacesInCStyleCastParentheses: false
SpacesInParentheses: false
Standard: c++20
EOF
    print_success "Created .clang-format configuration"
fi

# Create coverage configuration
print_info "Creating coverage configuration..."
cat > coverage.conf << 'EOF'
# Coverage configuration for FEM Core Library

# LCOV settings
genhtml_branch_coverage = 1
genhtml_function_coverage = 1
genhtml_legend = 1
genhtml_show_details = 1

# GCOV settings
gcov_tool = gcov
gcov_version = auto

# Exclusion patterns
exclude_pattern = /usr/*
exclude_pattern = */test/*
exclude_pattern = */tests/*
exclude_pattern = */_deps/*
exclude_pattern = */external/*
exclude_pattern = */CMakeFiles/*
EOF

# Create Git hooks directory
if [ -d ../.git ]; then
    print_info "Setting up Git hooks..."
    mkdir -p ../.git/hooks

    # Pre-commit hook for formatting
    cat > ../.git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for FEM Core Library

# Check if clang-format is available
if ! command -v clang-format >/dev/null 2>&1; then
    echo "Warning: clang-format not found, skipping format check"
    exit 0
fi

# Get list of changed files
files=$(git diff --cached --name-only --diff-filter=ACMR | grep -E '\.(cpp|hpp|h|cc|cxx)$')

if [ -z "$files" ]; then
    exit 0
fi

# Check formatting
format_issues=false
for file in $files; do
    if [ -f "$file" ]; then
        if ! clang-format -dry-run -Werror "$file" >/dev/null 2>&1; then
            echo "Format issue in: $file"
            format_issues=true
        fi
    fi
done

if [ "$format_issues" = true ]; then
    echo ""
    echo "Code formatting issues found. Run 'core-format' to fix them."
    echo "Or run: find . -name '*.h' -o -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i"
    exit 1
fi

exit 0
EOF
    chmod +x ../.git/hooks/pre-commit
    print_success "Created Git pre-commit hook for formatting"
fi

print_header "Installation Complete!"

print_success "Development tools have been installed and configured."
echo ""
print_info "To get started:"
echo "  1. Source the development environment: source scripts/dev-setup.sh"
echo "  2. Build the project: core-build"
echo "  3. Run tests: core-test"
echo "  4. Generate coverage: core-coverage"
echo ""
print_info "Available scripts:"
echo "  📊 scripts/coverage.sh              - Generate HTML coverage reports"
echo "  📋 scripts/detailed_line_coverage.sh - Line-by-line coverage analysis"
echo "  🔧 scripts/dev-setup.sh             - Load development aliases"
echo ""
print_warning "Don't forget to source dev-setup.sh in your shell profile for persistent aliases!"