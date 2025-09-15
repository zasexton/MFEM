# FEM Core Library Development Scripts

This directory contains development and analysis scripts for the FEM Core Library.

## Scripts Overview

### 📊 Coverage Analysis

#### `coverage.sh`
Generates comprehensive HTML coverage reports with smart test-to-header mapping.

**Features:**
- Auto-discovers tests and maps them to corresponding headers
- Generates overall and module-specific coverage reports
- Creates organized HTML structure mirroring test organization
- Provides detailed coverage statistics per module

**Usage:**
```bash
cd build
../scripts/coverage.sh
```

**Output:**
- `coverage_html/index.html` - Overall coverage report
- `coverage_html/unit/index.html` - Unit tests index
- `coverage_html/unit/{module}/index.html` - Module-specific reports
- `coverage_summary.txt` - Text summary
- `test_header_mapping.txt` - Test-to-header mappings

#### `detailed_line_coverage.sh`
Provides line-by-line coverage analysis for detailed code inspection.

**Features:**
- Line-by-line execution counts
- Identifies uncovered code sections
- Generates per-test detailed reports
- Shows execution frequency for each line

**Usage:**
```bash
cd build
../scripts/detailed_line_coverage.sh
```

**Output:**
- `detailed_line_reports/index.txt` - Master index
- `detailed_line_reports/{module}/module_summary.txt` - Module summaries
- `detailed_line_reports/{module}/{test}_line_coverage.txt` - Detailed reports

### 🔧 Development Setup

#### `install-dev-tools.sh`
Installs and configures development tools for the FEM Core Library.

**Installs:**
- Build tools (GCC/Clang, CMake)
- Coverage tools (lcov, gcovr)
- Analysis tools (Valgrind, Cppcheck, Clang-tidy)
- Documentation tools (Doxygen)

**Configures:**
- Development aliases and shortcuts
- Code formatting rules (.clang-format)
- Git hooks for code quality
- Coverage settings

**Usage:**
```bash
./scripts/install-dev-tools.sh
source scripts/dev-setup.sh  # Load development environment
```

## Quick Start

1. **Install development tools:**
   ```bash
   ./scripts/install-dev-tools.sh
   ```

2. **Load development environment:**
   ```bash
   source scripts/dev-setup.sh
   ```

3. **Build with coverage:**
   ```bash
   core-build
   core-test
   core-coverage
   ```

## Development Aliases

After sourcing `dev-setup.sh`, these aliases are available:

- `core-build` - Build the project with current configuration
- `core-test` - Run all unit tests
- `core-coverage` - Generate HTML coverage reports
- `core-detailed-coverage` - Generate detailed line coverage
- `core-clean` - Clean build directory
- `core-format` - Format code with clang-format
- `core-lint` - Run clang-tidy static analysis
- `core-check` - Run cppcheck static analysis

## Coverage Reports Structure

```
coverage_html/
├── index.html                 # Overall coverage
└── unit/
    ├── index.html             # Unit tests index
    ├── base/
    │   └── index.html         # Base module coverage
    ├── error/
    │   └── index.html         # Error module coverage
    └── ...
```

## Test-to-Header Mapping

The coverage scripts automatically map tests to their corresponding headers:

- `base_test_observer` → `observer.h`
- `base_test_policy` → `policy.h`
- `base_test_singleton` → `singleton.h`

This enables focused coverage analysis per component.

## Requirements

### System Requirements
- C++20 compatible compiler (GCC 11+, Clang 12+)
- CMake 3.20+
- Git

### Coverage Requirements
- lcov (for HTML reports)
- gcov (for line coverage data)
- gcovr (alternative coverage tool)

### Optional Tools
- Valgrind (memory analysis)
- Clang-format (code formatting)
- Clang-tidy (static analysis)
- Cppcheck (additional static analysis)
- Doxygen (documentation generation)

## Configuration

### Environment Variables
- `GCOV_TOOL` - Specify gcov version (default: gcov)
- `BUILD_DIR` - Build directory (default: build)
- `CMAKE_BUILD_TYPE` - Build type (default: Debug)

### Coverage Configuration
Coverage settings can be customized in `coverage.conf`:
- Exclusion patterns
- Report options
- Tool settings

## Troubleshooting

### No Coverage Data
1. Ensure tests are built with coverage flags
2. Run tests before generating coverage
3. Check that gcov files are generated

### Missing Test Mappings
1. Verify test naming follows `{module}_test_{component}` pattern
2. Check that header files exist in expected locations
3. Review mapping output in script logs

### Build Issues
1. Check C++20 compiler support
2. Verify CMake version compatibility
3. Ensure all dependencies are installed

## Contributing

When adding new tests or modules:

1. Follow naming convention: `{module}_test_{component}`
2. Place headers in appropriate module directories
3. Update scripts if new patterns are needed
4. Test coverage generation with new components

## Script Dependencies

The scripts use standard Unix tools:
- bash
- find
- grep
- sed
- awk
- bc (for percentage calculations)

All scripts are designed to work on Linux and macOS systems.