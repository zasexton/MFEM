#!/bin/bash
# detailed_line_coverage.sh - Line-by-line coverage for FEM Core Library

set -e

# Configuration
GCOV_TOOL=${GCOV_TOOL:-gcov-13}
BUILD_DIR=${BUILD_DIR:-build}
PROJECT_ROOT=${PROJECT_ROOT:-.}
HEADER_DIR="${PROJECT_ROOT}/base"
REPORT_DIR="detailed_line_reports"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Functions
print_header() {
    echo -e "\n${GREEN}=== $1 ===${NC}"
}

print_subheader() {
    echo -e "${YELLOW}>>> $1${NC}"
}

# Function to extract header name from test name
get_header_from_test() {
    local test_name=$1
    echo "$test_name" | sed 's/^[^_]*_test_//'
}

# Function to find header file
find_header_file() {
    local header_base=$1
    local module=$2

    local possible_paths=(
        "../${module}/${header_base}.h"
        "../${module}/${header_base}.hpp"
        "${HEADER_DIR}/${header_base}.h"
        "${HEADER_DIR}/${header_base}.hpp"
        "../base/${header_base}.h"
        "../base/${header_base}.hpp"
    )

    for path in "${possible_paths[@]}"; do
        if [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done

    return 1
}

# Function to get line coverage from gcov
get_line_coverage() {
    local header_file=$1
    local header_name=$(basename "$header_file")
    local header_base="${header_name%.*}"

    # Find gcov file
    local gcov_files=(
        "CMakeFiles/core.dir/${header_base}.gcov"
        "CMakeFiles/core.dir/base/${header_base}.gcov"
        "*/${header_base}.gcov"
        "*/*/${header_base}.gcov"
    )

    for pattern in "${gcov_files[@]}"; do
        for gcov_file in $pattern; do
            if [ -f "$gcov_file" ]; then
                echo "$gcov_file"
                return 0
            fi
        done
    done

    return 1
}

# Function to generate detailed line report
generate_line_report() {
    local test_name=$1
    local header_file=$2
    local module=$3
    local output_file="$REPORT_DIR/$module/${test_name}_line_coverage.txt"

    local header_name=$(basename "$header_file")

    {
        echo "=============================================="
        echo "Line-by-Line Coverage Report"
        echo "=============================================="
        echo "Test: $test_name"
        echo "Header: $header_name"
        echo "Path: $header_file"
        echo "Module: $module"
        echo "Generated: $(date)"
        echo "=============================================="
        echo ""

        # Try to find and parse gcov data
        if gcov_file=$(get_line_coverage "$header_file"); then
            echo "GCOV Data from: $gcov_file"
            echo "=============================================="

            # Parse gcov file
            local line_num=0
            local total_lines=0
            local covered_lines=0
            local executable_lines=0

            while IFS= read -r line; do
                line_num=$((line_num + 1))

                # Parse gcov line format: execution_count:line_number:source_code
                if [[ "$line" =~ ^([^:]*):([^:]*):(.*)$ ]]; then
                    local exec_count="${BASH_REMATCH[1]}"
                    local src_line_num="${BASH_REMATCH[2]}"
                    local source_code="${BASH_REMATCH[3]}"

                    # Clean up execution count
                    exec_count=$(echo "$exec_count" | xargs)

                    # Determine line status
                    local status=""
                    if [[ "$exec_count" == "-" ]]; then
                        status="    [NON-EXEC]"
                    elif [[ "$exec_count" == "#####" ]] || [[ "$exec_count" == "0" ]]; then
                        status="    [UNCOVERED]"
                        executable_lines=$((executable_lines + 1))
                    elif [[ "$exec_count" =~ ^[0-9]+$ ]]; then
                        status="    [COVERED:$exec_count]"
                        executable_lines=$((executable_lines + 1))
                        covered_lines=$((covered_lines + 1))
                    else
                        status="    [UNKNOWN]"
                    fi

                    printf "%4s:%4s: %-15s %s\n" "$exec_count" "$src_line_num" "$status" "$source_code"
                    total_lines=$((total_lines + 1))
                fi
            done < "$gcov_file"

            echo ""
            echo "=============================================="
            echo "Coverage Summary"
            echo "=============================================="
            echo "Total lines: $total_lines"
            echo "Executable lines: $executable_lines"
            echo "Covered lines: $covered_lines"
            if [ "$executable_lines" -gt 0 ]; then
                local coverage_percent=$(( (covered_lines * 100) / executable_lines ))
                echo "Line coverage: $coverage_percent% ($covered_lines/$executable_lines)"
            else
                echo "Line coverage: N/A (no executable lines)"
            fi

        else
            echo "GCOV data not found for $header_name"
            echo "=============================================="
            echo ""
            echo "This could mean:"
            echo "1. The header is template-only (no .cpp file)"
            echo "2. No coverage data was generated"
            echo "3. The test didn't exercise this header"
            echo ""
            echo "Trying to show header content for reference:"
            echo "=============================================="

            if [ -f "$header_file" ]; then
                local line_num=0
                while IFS= read -r line; do
                    line_num=$((line_num + 1))
                    printf "%4d: %s\n" "$line_num" "$line"
                done < "$header_file"
            else
                echo "Header file not accessible: $header_file"
            fi
        fi

    } > "$output_file"

    echo "    Generated: $output_file"
}

print_header "Smart Line-by-Line Coverage Analysis"

# Check if we're already in build directory
if [ "$(basename $(pwd))" = "build" ]; then
    echo "Already in build directory"
else
    cd "$BUILD_DIR"
fi

# Check for coverage data
if [ ! -f "coverage.filtered.info" ] && [ ! -f "coverage.info" ]; then
    echo -e "${RED}Error: No coverage data found!${NC}"
    echo "Please run ./coverage.sh first to generate coverage data."
    exit 1
fi

# Create report directory
print_subheader "Setting up report directories"
rm -rf "$REPORT_DIR"
mkdir -p "$REPORT_DIR"

# Discover test-to-header mappings
print_subheader "Building test-to-header mappings"
declare -A TEST_TO_HEADER_MAP
declare -A MODULE_HEADERS
TEST_MODULES=()

# Parse CTest output to build mappings
while IFS= read -r line; do
    if [[ "$line" =~ Test\ #[0-9]+:\ ([^[:space:]]+_test_[^[:space:]]+) ]]; then
        test_name="${BASH_REMATCH[1]}"

        if [[ "$test_name" =~ ^([^_]+)_test_ ]]; then
            module="${BASH_REMATCH[1]}"

            if [[ ! " ${TEST_MODULES[@]} " =~ " ${module} " ]]; then
                TEST_MODULES+=("$module")
                mkdir -p "$REPORT_DIR/$module"
            fi

            header_base=$(get_header_from_test "$test_name")

            if header_path=$(find_header_file "$header_base" "$module"); then
                TEST_TO_HEADER_MAP["$test_name"]="$header_path"

                if [ -z "${MODULE_HEADERS[$module]}" ]; then
                    MODULE_HEADERS[$module]="$header_path"
                else
                    MODULE_HEADERS[$module]="${MODULE_HEADERS[$module]}:$header_path"
                fi

                echo "  Mapped: $test_name -> $(basename $header_path)"
            fi
        fi
    fi
done < <(ctest -N 2>/dev/null)

# Run coverage with gcov
print_subheader "Running coverage analysis with gcov"
for module in "${TEST_MODULES[@]}"; do
    echo "Running tests for module: $module"
    ctest -R "^${module}_test_" --quiet || true
done

# Generate gcov files
print_subheader "Generating gcov files"
find . -name "*.gcno" -exec dirname {} \; | sort | uniq | while read dir; do
    if [ -d "$dir" ]; then
        echo "Processing gcov in: $dir"
        (cd "$dir" && $GCOV_TOOL *.gcno 2>/dev/null || true)
    fi
done

# Generate detailed reports for each test-header mapping
print_header "Generating Line-by-Line Reports"

for module in "${TEST_MODULES[@]}"; do
    print_subheader "Processing $module module"

    # Get all tests for this module
    while IFS= read -r line; do
        if [[ "$line" =~ Test\ #[0-9]+:\ (${module}_test_[^[:space:]]+) ]]; then
            test_name="${BASH_REMATCH[1]}"

            if [ -n "${TEST_TO_HEADER_MAP[$test_name]}" ]; then
                header_path="${TEST_TO_HEADER_MAP[$test_name]}"
                echo "  Processing: $test_name -> $(basename $header_path)"
                generate_line_report "$test_name" "$header_path" "$module"
            fi
        fi
    done < <(ctest -N 2>/dev/null)
done

# Generate module summary reports
print_header "Generating Module Summary Reports"

for module in "${TEST_MODULES[@]}"; do
    summary_file="$REPORT_DIR/$module/module_summary.txt"

    {
        echo "=============================================="
        echo "Module Coverage Summary: ${module^}"
        echo "=============================================="
        echo "Generated: $(date)"
        echo ""

        echo "Test-to-Header Mappings:"
        echo "------------------------"
        while IFS= read -r line; do
            if [[ "$line" =~ Test\ #[0-9]+:\ (${module}_test_[^[:space:]]+) ]]; then
                test_name="${BASH_REMATCH[1]}"
                if [ -n "${TEST_TO_HEADER_MAP[$test_name]}" ]; then
                    header_path="${TEST_TO_HEADER_MAP[$test_name]}"
                    printf "  %-30s -> %s\n" "$test_name" "$(basename $header_path)"
                fi
            fi
        done < <(ctest -N 2>/dev/null)

        echo ""
        echo "Available Reports:"
        echo "------------------"
        for report in "$REPORT_DIR/$module"/*_line_coverage.txt; do
            if [ -f "$report" ]; then
                echo "  $(basename $report)"
            fi
        done

        # Try to get overall module coverage from lcov
        if [ -f "../coverage_${module}.info" ]; then
            echo ""
            echo "Module Coverage Summary:"
            echo "------------------------"
            cd .. && lcov --summary "coverage_${module}.info" 2>/dev/null | grep -E "lines|functions" | sed 's/^/  /' && cd "$BUILD_DIR"
        fi

    } > "$summary_file"

    echo "Generated module summary: $summary_file"
done

# Generate master index
index_file="$REPORT_DIR/index.txt"
{
    echo "=============================================="
    echo "FEM Core Library - Detailed Line Coverage"
    echo "=============================================="
    echo "Generated: $(date)"
    echo ""

    echo "Available Modules:"
    echo "------------------"
    for module in "${TEST_MODULES[@]}"; do
        echo "  📂 $module/"
        echo "     ├── module_summary.txt"

        # List individual test reports
        for report in "$REPORT_DIR/$module"/*_line_coverage.txt; do
            if [ -f "$report" ]; then
                report_name=$(basename "$report")
                echo "     ├── $report_name"
            fi
        done
    done

    echo ""
    echo "Usage:"
    echo "------"
    echo "1. Start with module_summary.txt for an overview"
    echo "2. Review individual *_line_coverage.txt files for detailed analysis"
    echo "3. Look for [UNCOVERED] lines to identify untested code"
    echo "4. [COVERED:N] shows lines executed N times"
    echo ""

} > "$index_file"

# Print final summary
print_header "Line Coverage Analysis Complete!"

echo -e "${GREEN}Generated Reports:${NC}"
echo "  📄 Master Index:      $(pwd)/$index_file"
echo ""
for module in "${TEST_MODULES[@]}"; do
    echo "  📂 ${module^} Module:"
    echo "     📄 Summary:        $(pwd)/$REPORT_DIR/$module/module_summary.txt"

    report_count=$(ls "$REPORT_DIR/$module"/*_line_coverage.txt 2>/dev/null | wc -l)
    echo "     📊 Line Reports:   $report_count files"
done

echo ""
echo -e "${CYAN}How to Use:${NC}"
echo "1. Start with the master index: $REPORT_DIR/index.txt"
echo "2. Review module summaries for overview"
echo "3. Examine individual line coverage reports for details"
echo "4. Look for [UNCOVERED] lines to improve test coverage"

echo ""
echo -e "${YELLOW}Note:${NC} Line coverage for header-only templates may show"
echo "limited data since they're compiled inline with tests."