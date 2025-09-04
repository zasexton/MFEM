#!/bin/bash
set -e

# Configuration
GCOV_TOOL=${GCOV_TOOL:-gcov-13}
BUILD_DIR=${BUILD_DIR:-build}

cd $BUILD_DIR

echo "Cleaning previous coverage data..."
find . -name "*.gcda" -delete

echo "Running tests..."
ctest --output-on-failure --parallel $(nproc)

echo "Capturing coverage..."
lcov --capture \
     --directory . \
     --output-file coverage.info \
     --gcov-tool $GCOV_TOOL \
     --ignore-errors gcov

echo "Filtering coverage..."
lcov --remove coverage.info \
     '/usr/*' '*/test/*' '*/tests/*' '*/_deps/*' '*/external/*' \
     --output-file coverage.filtered.info \
     --gcov-tool $GCOV_TOOL

echo "Generating HTML report..."
genhtml coverage.filtered.info \
        --output-directory coverage_html \
        --title "FEM Numeric Coverage" \
        --legend --show-details \
        --ignore-errors source

echo "Coverage summary:"
lcov --summary coverage.filtered.info

echo -e "\n✓ Report generated at: $(pwd)/coverage_html/index.html"