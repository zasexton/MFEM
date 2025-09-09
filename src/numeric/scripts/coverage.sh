#!/bin/bash
# coverage.sh - Smart coverage generation with test-to-header mapping
# Uses test naming convention to automatically associate headers with tests

set -e

# Configuration
GCOV_TOOL=${GCOV_TOOL:-gcov-13}
BUILD_DIR=${BUILD_DIR:-build}
PROJECT_ROOT=${PROJECT_ROOT:-.}
INCLUDE_DIR="${PROJECT_ROOT}/include"

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
# unit_traits_test_type_traits -> type_traits
get_header_from_test() {
    local test_name=$1
    # Remove unit_<category>_test_ prefix to get the header base name
    echo "$test_name" | sed 's/^unit_[^_]*_test_//'
}

# Function to find header file from base name
find_header_file() {
    local header_base=$1
    local category=$2

    # Possible header locations and extensions
    local possible_paths=(
        "../include/${category}/${header_base}.h"
        "../include/${category}/${header_base}.hpp"
        "../include/traits/${header_base}.h"  # Special case for traits
        "../include/traits/${header_base}.hpp"
        "../include/${header_base}.h"
        "../include/${header_base}.hpp"
        "${INCLUDE_DIR}/${category}/${header_base}.h"
        "${INCLUDE_DIR}/${category}/${header_base}.hpp"
        "${INCLUDE_DIR}/traits/${header_base}.h"
        "${INCLUDE_DIR}/traits/${header_base}.hpp"
    )

    for path in "${possible_paths[@]}"; do
        if [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done

    return 1
}

print_header "FEM Numeric Smart Coverage Analysis"
echo "Build directory: $BUILD_DIR"
echo "Include directory: $INCLUDE_DIR"

cd $BUILD_DIR

# Clean previous coverage data
print_subheader "Cleaning previous coverage data"
find . -name "*.gcda" -delete 2>/dev/null || true
rm -f coverage*.info 2>/dev/null || true
rm -rf coverage*_html 2>/dev/null || true

# Discover tests and map to headers
print_subheader "Discovering tests and mapping to headers"
declare -A TEST_TO_HEADER_MAP
declare -A CATEGORY_HEADERS
TEST_CATEGORIES=()

# Get all test names from CTest
while IFS= read -r line; do
    if [[ "$line" =~ Test\ #[0-9]+:\ (unit_[^[:space:]]+) ]]; then
        test_name="${BASH_REMATCH[1]}"

        # Extract category from test name (unit_<category>_test_...)
        if [[ "$test_name" =~ ^unit_([^_]+)_test_ ]]; then
            category="${BASH_REMATCH[1]}"

            # Add category to list if not already there
            if [[ ! " ${TEST_CATEGORIES[@]} " =~ " ${category} " ]]; then
                TEST_CATEGORIES+=("$category")
            fi

            # Get header base name from test
            header_base=$(get_header_from_test "$test_name")

            # Find the actual header file
            if header_path=$(find_header_file "$header_base" "$category"); then
                TEST_TO_HEADER_MAP["$test_name"]="$header_path"

                # Add to category headers list
                if [ -z "${CATEGORY_HEADERS[$category]}" ]; then
                    CATEGORY_HEADERS[$category]="$header_path"
                else
                    CATEGORY_HEADERS[$category]="${CATEGORY_HEADERS[$category]}:$header_path"
                fi

                echo "  Mapped: $test_name -> $(basename $header_path)"
            else
                echo "  ${YELLOW}Warning: No header found for $test_name (looking for ${header_base}.h)${NC}"
            fi
        fi
    fi
done < <(ctest -N 2>/dev/null)

echo "Found ${#TEST_TO_HEADER_MAP[@]} test-to-header mappings"
echo "Categories: ${TEST_CATEGORIES[@]}"

# Run all tests first
print_header "Running All Tests"
ctest --output-on-failure --parallel $(nproc) -R "^unit_" || {
    echo -e "${YELLOW}Some tests failed, but continuing with coverage...${NC}"
}

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
        --title "FEM Numeric Coverage - All" \
        --legend \
        --show-details \
        --demangle-cpp \
        --ignore-errors source \
        --quiet

echo -e "${GREEN}✓${NC} Overall report: $(pwd)/coverage_html/index.html"

# Generate category-specific coverage with smart header mapping
print_header "Category-Specific Coverage (Smart Mapping)"

for category in "${TEST_CATEGORIES[@]}"; do
    print_subheader "Processing $category"

    # Get all headers for this category
    IFS=':' read -ra headers <<< "${CATEGORY_HEADERS[$category]}"

    if [ ${#headers[@]} -eq 0 ]; then
        echo -e "  ${YELLOW}No headers mapped for ${category}${NC}"
        continue
    fi

    echo "  Headers mapped for ${category}:"
    for header in "${headers[@]}"; do
        echo "    - $(basename $header)"
    done

    # Run tests for this category
    echo "  Running ${category} tests..."
    test_count=$(ctest -N -R "^unit_${category}_" 2>/dev/null | grep -c "Test #" || echo "0")
    echo "  Found $test_count ${category} tests"

    if [ "$test_count" -gt 0 ]; then
        # Run the tests
        ctest -R "^unit_${category}_" --quiet || true

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
                 --output-file coverage_${category}_raw.info \
                 --gcov-tool $GCOV_TOOL 2>/dev/null || {
                echo -e "  ${YELLOW}No coverage data extracted${NC}"
                continue
            }

            # Remove any test files that might have slipped in
            lcov --remove coverage_${category}_raw.info \
                 '/usr/*' '*/test/*' '*/tests/*' '*/_deps/*' \
                 --output-file coverage_${category}.info \
                 --gcov-tool $GCOV_TOOL 2>/dev/null

            # Check if we got data
            if [ -s coverage_${category}.info ]; then
                # Count covered files
                file_count=$(lcov --list coverage_${category}.info 2>/dev/null | grep -c "\.h" || echo "0")
                echo "  Coverage found for $file_count headers"

                # Show coverage for each mapped header
                echo "  Coverage by header:"
                for header in "${headers[@]}"; do
                    header_name=$(basename "$header")
                    coverage_line=$(lcov --list coverage_${category}.info 2>/dev/null | grep "$header_name" | head -1)
                    if [ -n "$coverage_line" ]; then
                        coverage=$(echo "$coverage_line" | awk -F'|' '{print $2}' | xargs)
                        echo "    - $header_name: $coverage"
                    else
                        echo "    - $header_name: No coverage data"
                    fi
                done

                # Generate HTML report
                genhtml coverage_${category}.info \
                        --output-directory coverage_${category}_html \
                        --title "FEM Numeric - ${category^} Coverage" \
                        --legend \
                        --show-details \
                        --demangle-cpp \
                        --ignore-errors source \
                        --quiet

                echo -e "  ${GREEN}✓${NC} ${category} report: $(pwd)/coverage_${category}_html/index.html"

                # Show summary
                echo "  Coverage summary:"
                lcov --summary coverage_${category}.info 2>/dev/null | grep -E "lines|functions" | sed 's/^/    /'
            else
                echo -e "  ${YELLOW}No coverage data after filtering${NC}"
            fi

            # Clean up
            rm -f coverage_${category}_raw.info
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

    for category in "${TEST_CATEGORIES[@]}"; do
        echo "${category^}:"
        echo "--------"

        # List each test and its associated header
        while IFS= read -r line; do
            if [[ "$line" =~ Test\ #[0-9]+:\ (unit_${category}_test_[^[:space:]]+) ]]; then
                test_name="${BASH_REMATCH[1]}"
                if [ -n "${TEST_TO_HEADER_MAP[$test_name]}" ]; then
                    header_path="${TEST_TO_HEADER_MAP[$test_name]}"
                    header_name=$(basename "$header_path")

                    # Try to get coverage for this header
                    coverage="N/A"
                    if [ -f "coverage_${category}.info" ]; then
                        coverage_line=$(lcov --list coverage_${category}.info 2>/dev/null | grep "$header_name" | head -1)
                        if [ -n "$coverage_line" ]; then
                            coverage=$(echo "$coverage_line" | awk -F'|' '{print $2}' | xargs)
                        fi
                    fi

                    printf "  %-40s -> %-25s [%s]\n" "$test_name" "$header_name" "$coverage"
                fi
            fi
        done < <(ctest -N 2>/dev/null)
        echo ""
    done
} > test_header_mapping.txt

echo "Test-to-header mapping saved to: test_header_mapping.txt"

# Generate summary report
print_header "Coverage Summary"

echo -e "\n${CYAN}Overall Coverage:${NC}"
lcov --summary coverage.filtered.info 2>/dev/null || echo "No data"

echo -e "\n${CYAN}Per-Category Coverage:${NC}"
for category in "${TEST_CATEGORIES[@]}"; do
    if [ -f "coverage_${category}.info" ] && [ -s "coverage_${category}.info" ]; then
        echo -e "\n${MAGENTA}${category^}:${NC}"
        lcov --summary coverage_${category}.info 2>/dev/null | grep -E "lines|functions" || echo "  No data"
    fi
done

# Create summary file
{
    echo "FEM Numeric Coverage Report"
    echo "Generated: $(date)"
    echo ""
    echo "Test Categories: ${TEST_CATEGORIES[@]}"
    echo ""
    echo "Test-to-Header Mappings:"
    for test_name in "${!TEST_TO_HEADER_MAP[@]}"; do
        echo "  $test_name -> $(basename ${TEST_TO_HEADER_MAP[$test_name]})"
    done
    echo ""
    echo "Overall Coverage:"
    lcov --summary coverage.filtered.info 2>/dev/null || echo "No data"
    echo ""
    for category in "${TEST_CATEGORIES[@]}"; do
        if [ -f "coverage_${category}.info" ]; then
            echo "${category^} Coverage:"
            lcov --summary coverage_${category}.info 2>/dev/null || echo "No data"
            echo ""
        fi
    done
} > coverage_summary.txt

# Print final report locations
print_header "Generated Reports"
echo -e "${GREEN}HTML Reports (Mirrored Structure):${NC}"
echo "  📊 Overall:           $(pwd)/coverage_html/index.html"
echo "  📁 Unit Tests Index:  $(pwd)/$COVERAGE_BASE_DIR/index.html"
for category in "${TEST_CATEGORIES[@]}"; do
    if [ -d "$COVERAGE_BASE_DIR/$category" ]; then
        echo "  📂 ${category^}:            $(pwd)/$COVERAGE_BASE_DIR/$category/index.html"
    fi
done

echo -e "\n${GREEN}HTML Reports (Legacy Locations):${NC}"
for category in "${TEST_CATEGORIES[@]}"; do
    if [ -d "coverage_${category}_html" ]; then
        echo "  📊 ${category^}:            $(pwd)/coverage_${category}_html/index.html"
    fi
done

echo -e "\n${GREEN}Text Reports:${NC}"
echo "  📄 Summary:           $(pwd)/coverage_summary.txt"
echo "  📄 Mapping:           $(pwd)/test_header_mapping.txt"

echo -e "\n${GREEN}Coverage Files:${NC}"
echo "  📁 Filtered:          $(pwd)/coverage.filtered.info"
for category in "${TEST_CATEGORIES[@]}"; do
    if [ -f "coverage_${category}.info" ]; then
        echo "  📁 ${category^}:             $(pwd)/coverage_${category}.info"
    fi
done

echo -e "\n${CYAN}Directory Structure:${NC}"
echo "  coverage_html/"
echo "  ├── index.html                    (Overall coverage)"
echo "  └── unit/"
echo "      ├── index.html                (Unit tests index)"
for category in "${TEST_CATEGORIES[@]}"; do
    echo "      ├── ${category}/"
    echo "      │   └── index.html            (${category^} coverage)"
done

print_header "Coverage Analysis Complete!"
echo "The coverage HTML structure now mirrors the test executable structure."
echo "Navigate to coverage_html/unit/ to see the organized reports."