#!/bin/bash
# detailed_line_coverage.sh

# Determine where we're running from
if [ -d "build" ]; then
    # Running from project root
    PROJECT_ROOT="."
    BUILD_DIR="build"
else
    # Running from somewhere else
    PROJECT_ROOT=".."
    BUILD_DIR="../build"
fi

REPORT_DIR="${BUILD_DIR}/detailed_line_reports"
GCOV_TOOL="gcov-13"

# Create the report directory
rm -rf ${REPORT_DIR}
mkdir -p ${REPORT_DIR}

# Work in build directory
cd ${BUILD_DIR}

# Check if coverage file exists
if [ ! -f coverage.filtered.info ]; then
    echo "Error: coverage.filtered.info not found. Run coverage generation first."
    exit 1
fi

echo "Starting line-by-line coverage report generation..."

# For each source file in coverage data
lcov --list coverage.filtered.info | grep -E "\.h|\.hpp" | while read line; do
    if [[ "$line" == *"|"* ]]; then
        filename=$(echo "$line" | cut -d'|' -f1 | xargs)

        echo "Generating line report for: $filename"

        # Find the source file
        source_file=""
        for dir in ../include ../src ../tests ..; do
            if [ -d "$dir" ]; then
                found=$(find $dir -name "$(basename $filename)" 2>/dev/null | head -1)
                if [ -f "$found" ]; then
                    source_file="$found"
                    break
                fi
            fi
        done

        if [ -f "$source_file" ]; then
            output_name=$(basename "$filename" | sed 's/[\/.]/_/g')_lines.txt
            # Use absolute path for output file
            output_file="$(pwd)/detailed_line_reports/${output_name}"

            echo "  Found source at: $source_file"
            echo "  Writing to: detailed_line_reports/${output_name}"

            # Extract coverage data for this file
            lcov --extract coverage.filtered.info "*$filename" \
                 --output-file temp_${output_name}.info \
                 --gcov-tool ${GCOV_TOOL} 2>/dev/null

            # Check if we got coverage data
            if [ -s temp_${output_name}.info ]; then
                # Parse coverage data into associative array
                declare -A line_hits

                while IFS= read -r info_line; do
                    if [[ "$info_line" == DA:* ]]; then
                        line_data="${info_line#DA:}"
                        line_num="${line_data%%,*}"
                        hit_count="${line_data#*,}"
                        line_hits[$line_num]=$hit_count
                    fi
                done < temp_${output_name}.info

                # Generate the report
                {
                    echo "LINE-BY-LINE COVERAGE: $filename"
                    echo "================================================================================"
                    echo ""
                    echo "Source file: $source_file"
                    echo "Coverage data: $(date)"
                    echo ""
                    echo "Legend:"
                    echo "  NOT EXEC = Line not executed"
                    echo "  <number> = Line executed <number> times"
                    echo "  -------- = Non-executable line"
                    echo ""
                    echo "================================================================================"
                    printf "%10s | %6s | %s\n" "Exec Count" "Line #" "Source Code"
                    echo "-----------|--------|----------------------------------------------------------"

                    # Read source file and annotate
                    line_no=1
                    while IFS= read -r source_line; do
                        # Truncate long lines
                        display_line="${source_line:0:70}"
                        [ ${#source_line} -gt 70 ] && display_line="${display_line}..."

                        if [[ -n "${line_hits[$line_no]}" ]]; then
                            hit_count="${line_hits[$line_no]}"
                            if [ "$hit_count" -eq 0 ]; then
                                printf "%10s | %6d | %s\n" "NOT EXEC" "$line_no" "$display_line"
                            else
                                printf "%10d | %6d | %s\n" "$hit_count" "$line_no" "$display_line"
                            fi
                        else
                            printf "%10s | %6d | %s\n" "--------" "$line_no" "$display_line"
                        fi
                        ((line_no++))
                    done < "$source_file"

                    echo ""
                    echo "================================================================================"
                    echo "Summary: ${#line_hits[@]} executable lines found in coverage data"

                } > "$output_file"

                # Clean up temp file
                rm -f temp_${output_name}.info

                echo "  ✓ Report saved successfully"
            else
                echo "  ⚠ No coverage data found for this file"
                echo "No coverage data available for $filename" > "$output_file"
            fi
        else
            echo "  ✗ Could not find source file"
        fi
    fi
done

# Check results
report_count=$(ls -1 detailed_line_reports/*.txt 2>/dev/null | wc -l)

echo ""
echo "==============================================="
echo "Line-by-line coverage report generation complete!"
echo "Total reports generated: $report_count"
echo "Location: ${BUILD_DIR}/detailed_line_reports/"
echo "==============================================="

if [ "$report_count" -gt 0 ]; then
    echo ""
    echo "Generated files:"
    ls -la detailed_line_reports/*.txt | head -10
fi