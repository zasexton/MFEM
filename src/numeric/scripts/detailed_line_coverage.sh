#!/bin/bash
# detailed_line_coverage.sh - Line-by-line coverage with smart test-to-header mapping

set -e

# Configuration
GCOV_TOOL=${GCOV_TOOL:-gcov-13}
BUILD_DIR=${BUILD_DIR:-build}
PROJECT_ROOT=${PROJECT_ROOT:-.}
INCLUDE_DIR="${PROJECT_ROOT}/include"
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
    echo "$test_name" | sed 's/^unit_[^_]*_test_//'
}

# Function to find header file
find_header_file() {
    local header_base=$1
    local category=$2

    local possible_paths=(
        "../include/${category}/${header_base}.h"
        "../include/${category}/${header_base}.hpp"
        "../include/traits/${header_base}.h"
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

print_header "Smart Line-by-Line Coverage Analysis"

cd "$BUILD_DIR"

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
declare -A CATEGORY_HEADERS
TEST_CATEGORIES=()

# Parse CTest output to build mappings
while IFS= read -r line; do
    if [[ "$line" =~ Test\ #[0-9]+:\ (unit_[^[:space:]]+) ]]; then
        test_name="${BASH_REMATCH[1]}"

        if [[ "$test_name" =~ ^unit_([^_]+)_test_ ]]; then
            category="${BASH_REMATCH[1]}"

            if [[ ! " ${TEST_CATEGORIES[@]} " =~ " ${category} " ]]; then
                TEST_CATEGORIES+=("$category")
                mkdir -p "$REPORT_DIR/$category"
            fi

            header_base=$(get_header_from_test "$test_name")

            if header_path=$(find_header_file "$header_base" "$category"); then
                TEST_TO_HEADER_MAP["$test_name"]="$header_path"

                # Track unique headers per category
                if [[ ! "${CATEGORY_HEADERS[$category]}" =~ "$header_path" ]]; then
                    if [ -z "${CATEGORY_HEADERS[$category]}" ]; then
                        CATEGORY_HEADERS[$category]="$header_path"
                    else
                        CATEGORY_HEADERS[$category]="${CATEGORY_HEADERS[$category]}:$header_path"
                    fi
                fi

                echo "  Mapped: $test_name -> $(basename $header_path)"
            fi
        fi
    fi
done < <(ctest -N 2>/dev/null)

echo "Found ${#TEST_TO_HEADER_MAP[@]} test-to-header mappings"
echo "Categories: ${TEST_CATEGORIES[@]}"

# Function to analyze a header with test association info
analyze_header_with_test_info() {
    local header_path=$1
    local category=$2
    local associated_test=$3
    local header_name=$(basename "$header_path")
    local output_file="$REPORT_DIR/$category/${header_name%.*}_lines.txt"

    echo -e "  ${BLUE}Analyzing: $header_name${NC}"
    echo "    Associated test: $associated_test"

    # Determine which coverage file to use
    local coverage_file="coverage.filtered.info"
    if [ -f "coverage_${category}.info" ]; then
        coverage_file="coverage_${category}.info"
    fi

    # Check if header is in coverage
    local header_in_coverage=$(lcov --list "$coverage_file" 2>/dev/null | grep -c "$header_name" || echo "0")

    if [ "$header_in_coverage" -eq 0 ]; then
        echo -e "    ${YELLOW}Not in coverage data${NC}"
        {
            echo "NO COVERAGE DATA AVAILABLE"
            echo ""
            echo "File: $header_name"
            echo "Category: $category"
            echo "Associated Test: $associated_test"
            echo ""
            echo "Possible reasons for missing coverage:"
            echo "  1. Header contains only templates that weren't instantiated"
            echo "  2. Test file doesn't actually use the header's code"
            echo "  3. Header contains only declarations"
            echo ""
            echo "To fix:"
            echo "  1. Ensure $associated_test actually uses code from $header_name"
            echo "  2. Add explicit template instantiations in the test"
            echo "  3. Verify the test is compiled with coverage flags"
        } > "$output_file"
        return
    fi

    # Extract coverage for this header
    local temp_info="$REPORT_DIR/temp_${header_name}.info"
    lcov --extract "$coverage_file" "*${header_name}" \
         --output-file "$temp_info" \
         --gcov-tool $GCOV_TOOL 2>/dev/null || {
        echo -e "    ${YELLOW}Extraction failed${NC}"
        return
    }

    # Parse coverage data
    declare -A line_hits
    declare -A function_hits
    local total_lines=0
    local covered_lines=0
    local total_functions=0
    local covered_functions=0

    while IFS= read -r line; do
        if [[ "$line" == DA:* ]]; then
            local data="${line#DA:}"
            local line_num="${data%%,*}"
            local hit_count="${data#*,}"
            line_hits[$line_num]=$hit_count
            ((total_lines++))
            [ "$hit_count" -gt 0 ] && ((covered_lines++))
        elif [[ "$line" == FN:* ]]; then
            local data="${line#FN:}"
            local line_num="${data%%,*}"
            local func_name="${data#*,}"
            function_hits[$func_name]=$line_num
            ((total_functions++))
        elif [[ "$line" == FNDA:* ]]; then
            local data="${line#FNDA:}"
            local hit_count="${data%%,*}"
            [ "$hit_count" -gt 0 ] && ((covered_functions++))
        fi
    done < "$temp_info"

    # Calculate percentages
    local line_coverage=0
    [ $total_lines -gt 0 ] && line_coverage=$((covered_lines * 100 / total_lines))

    local func_coverage=0
    [ $total_functions -gt 0 ] && func_coverage=$((covered_functions * 100 / total_functions))

    # Generate report
    {
        echo "================================================================================"
        echo "DETAILED LINE COVERAGE REPORT"
        echo "================================================================================"
        echo ""
        echo "Header File:     $header_name"
        echo "Full Path:       $header_path"
        echo "Category:        $category"
        echo "Tested By:       $associated_test"
        echo "Generated:       $(date)"
        echo ""
        echo "COVERAGE METRICS:"
        echo "  Lines:         $covered_lines / $total_lines ($line_coverage%)"
        echo "  Functions:     $covered_functions / $total_functions ($func_coverage%)"
        echo ""
        echo "TEST ASSOCIATION:"
        echo "  This header is tested by: $associated_test"
        echo "  Test executable: $(pwd)/tests/unit/${category}/${associated_test}"
        echo ""
        echo "LEGEND:"
        echo "  [EXEC n]    Line executed n times"
        echo "  [NOT EXEC]  Line not executed"
        echo "  [--------]  Non-executable line"
        echo ""
        echo "================================================================================"
        printf "%-12s | %6s | %s\n" "Status" "Line" "Source Code"
        echo "-------------|--------|-------------------------------------------------------"

        local line_no=1
        local uncovered_blocks=""
        local in_uncovered=false
        local block_start=0

        while IFS= read -r source_line; do
            local display="${source_line:0:65}"
            [ ${#source_line} -gt 65 ] && display="${display}..."
            display=$(echo "$display" | sed 's/^[[:space:]]*//')

            local status="[--------]"

            if [[ -n "${line_hits[$line_no]}" ]]; then
                local hits="${line_hits[$line_no]}"
                if [ "$hits" -eq 0 ]; then
                    status="[NOT EXEC]"
                    if [ "$in_uncovered" = false ]; then
                        block_start=$line_no
                        in_uncovered=true
                    fi
                else
                    status="[EXEC ${hits}x]"
                    [ ${#status} -lt 12 ] && status="$status "
                    if [ "$in_uncovered" = true ]; then
                        if [ $block_start -eq $((line_no-1)) ]; then
                            uncovered_blocks="${uncovered_blocks}  Line $block_start\n"
                        else
                            uncovered_blocks="${uncovered_blocks}  Lines $block_start-$((line_no-1))\n"
                        fi
                        in_uncovered=false
                    fi
                fi
            else
                if [ "$in_uncovered" = true ]; then
                    if [ $block_start -eq $((line_no-1)) ]; then
                        uncovered_blocks="${uncovered_blocks}  Line $block_start\n"
                    else
                        uncovered_blocks="${uncovered_blocks}  Lines $block_start-$((line_no-1))\n"
                    fi
                    in_uncovered=false
                fi
            fi

            printf "%-12s | %6d | %s\n" "$status" "$line_no" "$display"
            ((line_no++))
        done < "$header_path"

        if [ "$in_uncovered" = true ]; then
            if [ $block_start -eq $((line_no-1)) ]; then
                uncovered_blocks="${uncovered_blocks}  Line $block_start\n"
            else
                uncovered_blocks="${uncovered_blocks}  Lines $block_start-$((line_no-1))\n"
            fi
        fi

        echo ""
        echo "================================================================================"
        echo "ANALYSIS SUMMARY:"
        echo ""
        echo "Coverage Statistics:"
        echo "  Total executable lines:     $total_lines"
        echo "  Lines executed:            $covered_lines"
        echo "  Lines not executed:        $((total_lines - covered_lines))"
        echo "  Line coverage:             $line_coverage%"

        if [ $total_functions -gt 0 ]; then
            echo ""
            echo "Function Coverage:"
            echo "  Total functions:           $total_functions"
            echo "  Functions called:          $covered_functions"
            echo "  Functions not called:      $((total_functions - covered_functions))"
            echo "  Function coverage:         $func_coverage%"
        fi

        if [ -n "$uncovered_blocks" ]; then
            echo ""
            echo "Uncovered Code Blocks:"
            echo -e "$uncovered_blocks"
        fi

        echo ""
        echo "Coverage Grade: $(
            if [ $line_coverage -ge 90 ]; then echo "A (Excellent)"
            elif [ $line_coverage -ge 80 ]; then echo "B (Good)"
            elif [ $line_coverage -ge 70 ]; then echo "C (Fair)"
            elif [ $line_coverage -ge 60 ]; then echo "D (Poor)"
            else echo "F (Needs Improvement)"
            fi
        )"

        echo ""
        echo "TO IMPROVE COVERAGE:"
        echo "  1. Review uncovered lines above"
        echo "  2. Add test cases in $associated_test"
        echo "  3. Ensure template instantiations if header is template-heavy"
        echo "================================================================================"
    } > "$output_file"

    rm -f "$temp_info"

    if [ $line_coverage -ge 80 ]; then
        echo -e "    ${GREEN}✓ Coverage: $line_coverage% (Good)${NC}"
    elif [ $line_coverage -ge 60 ]; then
        echo -e "    ${YELLOW}⚠ Coverage: $line_coverage% (Fair)${NC}"
    else
        echo -e "    ${RED}✗ Coverage: $line_coverage% (Low)${NC}"
    fi
}

# Process headers by category using smart mapping
print_header "Generating Line Reports with Test Associations"

for category in "${TEST_CATEGORIES[@]}"; do
    print_subheader "Category: $category"

    # Get unique headers for this category
    IFS=':' read -ra headers <<< "${CATEGORY_HEADERS[$category]}"

    if [ ${#headers[@]} -eq 0 ]; then
        echo -e "  ${YELLOW}No headers mapped for ${category}${NC}"
        continue
    fi

    # Process each header
    for header_path in "${headers[@]}"; do
        # Find the test associated with this header
        associated_test=""
        for test_name in "${!TEST_TO_HEADER_MAP[@]}"; do
            if [ "${TEST_TO_HEADER_MAP[$test_name]}" = "$header_path" ]; then
                associated_test="$test_name"
                break
            fi
        done

        if [ -n "$associated_test" ]; then
            analyze_header_with_test_info "$header_path" "$category" "$associated_test"
        else
            echo -e "  ${YELLOW}Warning: No test found for $(basename $header_path)${NC}"
        fi
    done
done

# Generate enhanced index with test associations
print_header "Generating Enhanced Index"

index_file="$REPORT_DIR/INDEX.html"
{
    echo "<!DOCTYPE html>"
    echo "<html><head>"
    echo "<title>Line Coverage with Test Associations</title>"
    echo "<style>"
    echo "body { font-family: 'Segoe UI', Tahoma, sans-serif; margin: 20px; background: #f5f5f5; }"
    echo "h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }"
    echo "h2 { color: #666; margin-top: 30px; background: #e0e0e0; padding: 10px; }"
    echo ".category { margin: 20px 0; background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }"
    echo ".header-entry { margin: 10px 20px; padding: 10px; background: #fafafa; border-left: 4px solid #ddd; }"
    echo ".good { border-left-color: #4CAF50; }"
    echo ".fair { border-left-color: #FF9800; }"
    echo ".poor { border-left-color: #f44336; }"
    echo ".none { border-left-color: #9E9E9E; }"
    echo ".header-name { font-weight: bold; font-size: 1.1em; }"
    echo ".test-name { color: #666; font-size: 0.9em; margin-top: 5px; }"
    echo ".coverage { float: right; font-weight: bold; padding: 2px 8px; border-radius: 3px; }"
    echo ".coverage.good { background: #4CAF50; color: white; }"
    echo ".coverage.fair { background: #FF9800; color: white; }"
    echo ".coverage.poor { background: #f44336; color: white; }"
    echo ".coverage.none { background: #9E9E9E; color: white; }"
    echo "a { text-decoration: none; color: inherit; }"
    echo "</style>"
    echo "</head><body>"
    echo "<h1>📊 Line Coverage Reports with Test Associations</h1>"
    echo "<p>Generated: $(date)</p>"

    for category in "${TEST_CATEGORIES[@]}"; do
        echo "<div class='category'>"
        echo "<h2>📁 ${category^}</h2>"

        for report in "$REPORT_DIR/$category"/*.txt; do
            if [ -f "$report" ]; then
                local name=$(basename "$report" .txt)
                local header_name="${name%_lines}.h"
                local coverage="N/A"
                local class="none"
                local associated_test=""

                # Find associated test
                for test_name in "${!TEST_TO_HEADER_MAP[@]}"; do
                    if [[ "${TEST_TO_HEADER_MAP[$test_name]}" =~ "$header_name" ]]; then
                        associated_test="$test_name"
                        break
                    fi
                done

                if grep -q "NO COVERAGE DATA" "$report"; then
                    coverage="No data"
                else
                    coverage=$(grep "Line coverage:" "$report" | grep -o '[0-9]*%' || echo "N/A")
                    local cov_num=$(echo "$coverage" | tr -d '%')
                    if [ "$cov_num" -ge 80 ]; then class="good"
                    elif [ "$cov_num" -ge 60 ]; then class="fair"
                    else class="poor"
                    fi
                fi

                echo "<div class='header-entry $class'>"
                echo "  <a href='$category/$(basename "$report")'>"
                echo "    <span class='coverage $class'>$coverage</span>"
                echo "    <div class='header-name'>📄 $name</div>"
                if [ -n "$associated_test" ]; then
                    echo "    <div class='test-name'>🧪 Tested by: $associated_test</div>"
                else
                    echo "    <div class='test-name'>⚠️ No associated test found</div>"
                fi
                echo "  </a>"
                echo "</div>"
            fi
        done
        echo "</div>"
    done

    echo "</body></html>"
} > "$index_file"

# Print summary
print_header "Summary Statistics"

total_headers=0
headers_with_tests=0
headers_with_coverage=0

for test in "${!TEST_TO_HEADER_MAP[@]}"; do
    ((headers_with_tests++))
done

for category in "${TEST_CATEGORIES[@]}"; do
    for report in "$REPORT_DIR/$category"/*.txt; do
        if [ -f "$report" ]; then
            ((total_headers++))
            if ! grep -q "NO COVERAGE DATA" "$report"; then
                ((headers_with_coverage++))
            fi
        fi
    done
done

echo "Total headers analyzed:          $total_headers"
echo "Headers with associated tests:   $headers_with_tests"
echo "Headers with coverage data:      $headers_with_coverage"
echo "Headers without coverage:        $((total_headers - headers_with_coverage))"

print_header "Generated Reports"
echo -e "${GREEN}Report Locations:${NC}"
echo "  📁 Report Directory:    $(pwd)/$REPORT_DIR/"
echo "  🌐 HTML Index:          $(pwd)/$REPORT_DIR/INDEX.html"
echo ""
echo "The reports now show which test file is responsible for testing each header."

print_header "Smart Coverage Analysis Complete!"