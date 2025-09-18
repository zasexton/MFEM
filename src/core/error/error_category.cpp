#include "error_category.h"

namespace fem::core::error {

// NumericErrorCategory implementation
std::string NumericErrorCategory::message(int code) const {
    switch (code) {
        case 0: return "Success";
        case 1: return "Division by zero";
        case 2: return "Overflow error";
        case 3: return "Underflow error";
        case 4: return "Invalid operation";
        case 5: return "Convergence failed";
        case 6: return "Matrix singular";
        case 7: return "Invalid dimension";
        case 8: return "Numerical instability";
        default: return "Unknown numeric error";
    }
}

bool NumericErrorCategory::is_recoverable(int code) const noexcept {
    switch (code) {
        case 0:  // Success
        case 5:  // Convergence failed - can retry with different parameters
        case 8:  // Numerical instability - can try different algorithm
            return true;
        default:
            return false;
    }
}

int NumericErrorCategory::severity(int code) const noexcept {
    switch (code) {
        case 0: return 0;  // Info - success
        case 5: return 1;  // Warning - convergence
        case 8: return 1;  // Warning - instability
        case 1: return 3;  // Fatal - division by zero
        case 6: return 3;  // Fatal - singular matrix
        default: return 2; // Error - default
    }
}

// IOErrorCategory implementation
std::string IOErrorCategory::message(int code) const {
    switch (code) {
        case 0: return "Success";
        case 1: return "File not found";
        case 2: return "Permission denied";
        case 3: return "File already exists";
        case 4: return "Invalid file format";
        case 5: return "Disk full";
        case 6: return "Read error";
        case 7: return "Write error";
        case 8: return "End of file";
        case 9: return "Invalid path";
        case 10: return "Too many open files";
        default: return "Unknown I/O error";
    }
}

bool IOErrorCategory::is_recoverable(int code) const noexcept {
    switch (code) {
        case 0:  // Success
        case 5:  // Disk full - might free up space
        case 10: // Too many open files - can close some
            return true;
        default:
            return false;
    }
}

std::string IOErrorCategory::suggested_action(int code) const {
    switch (code) {
        case 1: return "Check if file exists and path is correct";
        case 2: return "Check file permissions";
        case 3: return "Use a different filename or delete existing file";
        case 4: return "Verify file format is supported";
        case 5: return "Free up disk space";
        case 6:
        case 7: return "Check disk health and file system integrity";
        case 9: return "Verify path is valid for this operating system";
        case 10: return "Close unused file handles";
        default: return "Check system logs for more details";
    }
}

// MemoryErrorCategory implementation
std::string MemoryErrorCategory::message(int code) const {
    switch (code) {
        case 0: return "Success";
        case 1: return "Out of memory";
        case 2: return "Invalid pointer";
        case 3: return "Double free";
        case 4: return "Memory leak detected";
        case 5: return "Buffer overflow";
        case 6: return "Buffer underflow";
        case 7: return "Use after free";
        case 8: return "Alignment error";
        case 9: return "Memory corruption";
        default: return "Unknown memory error";
    }
}

int MemoryErrorCategory::severity(int code) const noexcept {
    switch (code) {
        case 0: return 0;  // Info - success
        case 4: return 1;  // Warning - memory leak
        case 1: return 2;  // Error - out of memory
        default: return 3; // Fatal - corruption/overflow/etc
    }
}

// Global category accessors
const NumericErrorCategory& numeric_category() noexcept {
    static NumericErrorCategory instance;
    return instance;
}

const IOErrorCategory& io_category() noexcept {
    static IOErrorCategory instance;
    return instance;
}

const MemoryErrorCategory& memory_category() noexcept {
    static MemoryErrorCategory instance;
    return instance;
}

} // namespace fem::core::error