#!/bin/bash
# coverage.sh - Smart coverage generation for FEM Core Library
# Uses test naming convention to automatically associate headers with tests

set -e

# Configuration
GCOV_TOOL=${GCOV_TOOL:-gcov-13}
BUILD_DIR=${BUILD_DIR:-build}
PROJECT_ROOT=${PROJECT_ROOT:-.}
HEADER_DIR="${PROJECT_ROOT}/base"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Function to print section headers
print_header() {
    echo -e "\n${GREEN}=== $1 ===${NC}"
}

print_subheader() {
    echo -e "${YELLOW}>>> $1${NC}"
}

# Function to extract header name from test name
# test_observer -> observer
get_header_from_test() {
    local test_name=$1
    # Remove test_ prefix to get the header base name
    echo "$test_name" | sed 's/^test_//'
}

# Function to find header file from base name
find_header_file() {
    local header_base=$1
    local module=$2

    # Possible header locations and extensions
    local possible_paths=(
        "../${module}/${header_base}.h"
        "../${module}/${header_base}.hpp"
        "../base/${header_base}.h"
        "../base/${header_base}.hpp"
        "../../${module}/${header_base}.h"
        "../../${module}/${header_base}.hpp"
        "../../base/${header_base}.h"
        "../../base/${header_base}.hpp"
    )

    for path in "${possible_paths[@]}"; do
        if [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done

    return 1
}

print_header "FEM Core Library Smart Coverage Analysis"
echo "Build directory: $BUILD_DIR"
echo "Header directory: $HEADER_DIR"

# Check if we're already in build directory
if [ "$(basename $(pwd))" = "build" ]; then
    echo "Already in build directory"
else
    cd $BUILD_DIR
fi

# Clean previous coverage data
print_subheader "Cleaning previous coverage data"
find . -name "*.gcda" -delete 2>/dev/null || true
rm -f coverage*.info 2>/dev/null || true
rm -rf coverage*_html 2>/dev/null || true

# Discover tests and map to headers
print_subheader "Discovering tests and mapping to headers"
declare -A TEST_TO_HEADER_MAP
declare -A MODULE_HEADERS
TEST_MODULES=()

# Get all test executables directly from filesystem
echo "  Looking for test executables in tests/unit/*/test_*"
for test_executable in tests/unit/*/test_*; do
    echo "  Checking: $test_executable"
    if [ -f "$test_executable" ]; then
        test_name=$(basename "$test_executable")
        echo "    Found test: $test_name"

        # Extract component from test name (test_component -> component)
        if [[ "$test_name" =~ ^test_(.+) ]]; then
            header_base="${BASH_REMATCH[1]}"

            # For core library, we determine the module from the directory structure
            test_dir=$(dirname "$test_executable")
            module=$(basename "$test_dir")

            echo "    Header base: $header_base"
            echo "    Module: $module"

            # Add module to list if not already there
            if [[ ! " ${TEST_MODULES[@]} " =~ " ${module} " ]]; then
                TEST_MODULES+=("$module")
            fi

            # Find the actual header file
            if header_path=$(find_header_file "$header_base" "$module"); then
                TEST_TO_HEADER_MAP["$test_name"]="$header_path"

                # Add to module headers list
                if [ -z "${MODULE_HEADERS[$module]}" ]; then
                    MODULE_HEADERS[$module]="$header_path"
                else
                    MODULE_HEADERS[$module]="${MODULE_HEADERS[$module]}:$header_path"
                fi

                echo "  Mapped: $test_name -> $(basename $header_path)"
            else
                echo "  ${YELLOW}Warning: No header found for $test_name (looking for ${header_base}.h)${NC}"
            fi
        else
            echo "    No pattern match for: $test_name"
        fi
    else
        echo "    Not a file: $test_executable"
    fi
done

echo "Found ${#TEST_TO_HEADER_MAP[@]} test-to-header mappings"
echo "Modules: ${TEST_MODULES[@]}"

# Run all tests first
print_header "Running All Tests"
for test_executable in tests/unit/*/test_*; do
    if [ -f "$test_executable" ]; then
        echo "Running $(basename $test_executable)..."
        "./$test_executable" --gtest_brief=1 || {
            echo -e "${YELLOW}Test $(basename $test_executable) failed, but continuing...${NC}"
        }
    fi
done

# Capture overall coverage
print_header "Capturing Coverage Data"
print_subheader "Generating coverage.info"
lcov --capture \
     --directory . \
     --output-file coverage.info \
     --gcov-tool $GCOV_TOOL \
     --ignore-errors gcov

# Filter out system and test files
print_subheader "Filtering coverage data"
lcov --remove coverage.info \
     '/usr/*' \
     '*/test/*' \
     '*/tests/unit/*' \
     '*/_deps/*' \
     '*/external/*' \
     '*/CMakeFiles/*' \
     --output-file coverage.filtered.info \
     --gcov-tool $GCOV_TOOL

# Generate overall HTML report
print_subheader "Generating overall HTML report"
genhtml coverage.filtered.info \
        --output-directory coverage_html \
        --title "FEM Core Library Coverage - All" \
        --legend \
        --show-details \
        --demangle-cpp \
        --ignore-errors source \
        --quiet

echo -e "${GREEN}✓${NC} Overall report: $(pwd)/coverage_html/index.html"

# Generate module-specific coverage with smart header mapping
print_header "Module-Specific Coverage (Smart Mapping)"

for module in "${TEST_MODULES[@]}"; do
    print_subheader "Processing $module"

    # Get all headers for this module
    IFS=':' read -ra headers <<< "${MODULE_HEADERS[$module]}"

    if [ ${#headers[@]} -eq 0 ]; then
        echo -e "  ${YELLOW}No headers mapped for ${module}${NC}"
        continue
    fi

    echo "  Headers mapped for ${module}:"
    for header in "${headers[@]}"; do
        echo "    - $(basename $header)"
    done

    # Run tests for this module
    echo "  Running ${module} tests..."
    test_count=0
    for test_executable in tests/unit/${module}/test_*; do
        if [ -f "$test_executable" ]; then
            test_name=$(basename "$test_executable")
            test_count=$((test_count + 1))
            echo "    Running $test_name..."
            "./$test_executable" --gtest_brief=1 >/dev/null 2>&1 || true
        fi
    done
    echo "  Found $test_count ${module} tests"

    if [ "$test_count" -gt 0 ]; then

        # Create extraction patterns from mapped headers
        extraction_patterns=()
        for header in "${headers[@]}"; do
            # Add both the full path and just the filename pattern
            header_name=$(basename "$header")
            extraction_patterns+=("*${header_name}")

            # Also add the directory pattern
            header_dir=$(dirname "$header")
            header_dir_name=$(basename "$header_dir")
            extraction_patterns+=("*/${header_dir_name}/${header_name}")
        done

        echo "  Extracting coverage for mapped headers..."

        # Extract coverage for the specific headers
        if [ ${#extraction_patterns[@]} -gt 0 ]; then
            lcov --extract coverage.info \
                 "${extraction_patterns[@]}" \
                 --output-file coverage_${module}_raw.info \
                 --gcov-tool $GCOV_TOOL 2>/dev/null || {
                echo -e "  ${YELLOW}No coverage data extracted${NC}"
                continue
            }

            # Remove any test files that might have slipped in
            lcov --remove coverage_${module}_raw.info \
                 '/usr/*' '*/test/*' '*/tests/*' '*/_deps/*' \
                 --output-file coverage_${module}.info \
                 --gcov-tool $GCOV_TOOL 2>/dev/null

            # Check if we got data
            if [ -s coverage_${module}.info ]; then
                # Count covered files
                file_count=$(lcov --list coverage_${module}.info 2>/dev/null | grep -c "\.h" || echo "0")
                echo "  Coverage found for $file_count headers"

                # Show coverage for each mapped header
                echo "  Coverage by header:"
                for header in "${headers[@]}"; do
                    header_name=$(basename "$header")
                    coverage_line=$(lcov --list coverage_${module}.info 2>/dev/null | grep "$header_name" | head -1)
                    if [ -n "$coverage_line" ]; then
                        coverage=$(echo "$coverage_line" | awk -F'|' '{print $2}' | xargs)
                        echo "    - $header_name: $coverage"
                    else
                        echo "    - $header_name: No coverage data"
                    fi
                done

                # Generate HTML report
                genhtml coverage_${module}.info \
                        --output-directory coverage_${module}_html \
                        --title "FEM Core - ${module^} Coverage" \
                        --legend \
                        --show-details \
                        --demangle-cpp \
                        --ignore-errors source \
                        --quiet

                echo -e "  ${GREEN}✓${NC} ${module} report: $(pwd)/coverage_${module}_html/index.html"

                # Show summary
                echo "  Coverage summary:"
                lcov --summary coverage_${module}.info 2>/dev/null | grep -E "lines|functions" | sed 's/^/    /'
            else
                echo -e "  ${YELLOW}No coverage data after filtering${NC}"
            fi

            # Clean up
            rm -f coverage_${module}_raw.info
        fi
    fi
done

# Generate test-to-header mapping report
print_header "Test-to-Header Mapping Report"
{
    echo "Test-to-Header Coverage Mapping"
    echo "================================"
    echo "Generated: $(date)"
    echo ""

    for module in "${TEST_MODULES[@]}"; do
        echo "${module^}:"
        echo "--------"

        # List each test and its associated header for this module
        for test_executable in tests/unit/${module}/test_*; do
            if [ -f "$test_executable" ]; then
                test_name=$(basename "$test_executable")
                if [ -n "${TEST_TO_HEADER_MAP[$test_name]}" ]; then
                    header_path="${TEST_TO_HEADER_MAP[$test_name]}"
                    header_name=$(basename "$header_path")

                    # Try to get coverage for this header
                    coverage="N/A"
                    if [ -f "coverage_${module}.info" ]; then
                        coverage_line=$(lcov --list coverage_${module}.info 2>/dev/null | grep "$header_name" | head -1)
                        if [ -n "$coverage_line" ]; then
                            coverage=$(echo "$coverage_line" | awk -F'|' '{print $2}' | xargs)
                        fi
                    fi

                    printf "  %-40s -> %-25s [%s]\n" "$test_name" "$header_name" "$coverage"
                fi
            fi
        done
        echo ""
    done
} > test_header_mapping.txt

echo "Test-to-header mapping saved to: test_header_mapping.txt"

# Create organized HTML structure
print_subheader "Creating organized HTML structure"
COVERAGE_BASE_DIR="coverage_html/unit"
mkdir -p "$COVERAGE_BASE_DIR"

# Create index for unit tests
{
    echo '<!DOCTYPE html>'
    echo '<html><head><title>FEM Core Unit Tests Coverage</title>'
    echo '<style>'
    echo 'body { font-family: Arial, sans-serif; margin: 40px; }'
    echo 'h1 { color: #2c3e50; }'
    echo 'h2 { color: #34495e; margin-top: 30px; }'
    echo '.module { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }'
    echo '.module h3 { margin-top: 0; color: #2980b9; }'
    echo 'a { color: #3498db; text-decoration: none; }'
    echo 'a:hover { text-decoration: underline; }'
    echo '.coverage { font-weight: bold; }'
    echo '.high { color: #27ae60; }'
    echo '.medium { color: #f39c12; }'
    echo '.low { color: #e74c3c; }'
    echo '</style>'
    echo '</head><body>'
    echo '<h1>FEM Core Library - Unit Tests Coverage</h1>'
    echo '<p>Generated: <strong>'$(date)'</strong></p>'
    echo '<h2>Overall Coverage</h2>'
    echo '<p><a href="../index.html">📊 Overall Coverage Report</a></p>'
    echo '<h2>Module Coverage Reports</h2>'

    for module in "${TEST_MODULES[@]}"; do
        if [ -f "coverage_${module}.info" ] && [ -s "coverage_${module}.info" ]; then
            # Copy module coverage to organized structure
            cp -r "coverage_${module}_html" "$COVERAGE_BASE_DIR/$module" 2>/dev/null || true

            echo '<div class="module">'
            echo "<h3>📂 ${module^} Module</h3>"
            echo "<p><a href=\"$module/index.html\">View ${module^} Coverage Report</a></p>"

            # Get coverage summary
            if summary=$(lcov --summary coverage_${module}.info 2>/dev/null | grep "lines" | head -1); then
                percentage=$(echo "$summary" | grep -o '[0-9.]*%' | head -1)
                if [ -n "$percentage" ]; then
                    percent_num=$(echo "$percentage" | sed 's/%//')
                    if [ "$(echo "$percent_num >= 80" | bc -l 2>/dev/null)" = "1" ]; then
                        class="high"
                    elif [ "$(echo "$percent_num >= 60" | bc -l 2>/dev/null)" = "1" ]; then
                        class="medium"
                    else
                        class="low"
                    fi
                    echo "<p>Line Coverage: <span class=\"coverage $class\">$percentage</span></p>"
                fi
            fi
            echo '</div>'
        fi
    done

    echo '</body></html>'
} > "$COVERAGE_BASE_DIR/index.html"

# Generate summary report
print_header "Coverage Summary"

echo -e "\n${CYAN}Overall Coverage:${NC}"
lcov --summary coverage.filtered.info 2>/dev/null || echo "No data"

echo -e "\n${CYAN}Per-Module Coverage:${NC}"
for module in "${TEST_MODULES[@]}"; do
    if [ -f "coverage_${module}.info" ] && [ -s "coverage_${module}.info" ]; then
        echo -e "\n${MAGENTA}${module^}:${NC}"
        lcov --summary coverage_${module}.info 2>/dev/null | grep -E "lines|functions" || echo "  No data"
    fi
done

# Create summary file
{
    echo "FEM Core Library Coverage Report"
    echo "Generated: $(date)"
    echo ""
    echo "Test Modules: ${TEST_MODULES[@]}"
    echo ""
    echo "Test-to-Header Mappings:"
    for test_name in "${!TEST_TO_HEADER_MAP[@]}"; do
        echo "  $test_name -> $(basename ${TEST_TO_HEADER_MAP[$test_name]})"
    done
    echo ""
    echo "Overall Coverage:"
    lcov --summary coverage.filtered.info 2>/dev/null || echo "No data"
    echo ""
    for module in "${TEST_MODULES[@]}"; do
        if [ -f "coverage_${module}.info" ]; then
            echo "${module^} Coverage:"
            lcov --summary coverage_${module}.info 2>/dev/null || echo "No data"
            echo ""
        fi
    done
} > coverage_summary.txt

# Print final report locations
print_header "Generated Reports"
echo -e "${GREEN}HTML Reports (Organized Structure):${NC}"
echo "  📊 Overall:           $(pwd)/coverage_html/index.html"
echo "  📁 Unit Tests Index:  $(pwd)/$COVERAGE_BASE_DIR/index.html"
for module in "${TEST_MODULES[@]}"; do
    if [ -d "$COVERAGE_BASE_DIR/$module" ]; then
        echo "  📂 ${module^}:            $(pwd)/$COVERAGE_BASE_DIR/$module/index.html"
    fi
done

echo -e "\n${GREEN}HTML Reports (Individual):${NC}"
for module in "${TEST_MODULES[@]}"; do
    if [ -d "coverage_${module}_html" ]; then
        echo "  📊 ${module^}:            $(pwd)/coverage_${module}_html/index.html"
    fi
done

echo -e "\n${GREEN}Text Reports:${NC}"
echo "  📄 Summary:           $(pwd)/coverage_summary.txt"
echo "  📄 Mapping:           $(pwd)/test_header_mapping.txt"

echo -e "\n${GREEN}Coverage Files:${NC}"
echo "  📁 Filtered:          $(pwd)/coverage.filtered.info"
for module in "${TEST_MODULES[@]}"; do
    if [ -f "coverage_${module}.info" ]; then
        echo "  📁 ${module^}:             $(pwd)/coverage_${module}.info"
    fi
done

echo -e "\n${CYAN}Directory Structure:${NC}"
echo "  coverage_html/"
echo "  ├── index.html                    (Overall coverage)"
echo "  └── unit/"
echo "      ├── index.html                (Unit tests index)"
for module in "${TEST_MODULES[@]}"; do
    if [ -d "$COVERAGE_BASE_DIR/$module" ]; then
        echo "      ├── ${module}/"
        echo "      │   └── index.html            (${module^} coverage)"
    fi
done

print_header "Coverage Analysis Complete!"
echo "The coverage HTML structure now mirrors the test executable structure."
echo "Navigate to coverage_html/unit/ to see the organized reports."