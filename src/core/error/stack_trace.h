#pragma once

#ifndef CORE_ERROR_STACK_TRACE_H
#define CORE_ERROR_STACK_TRACE_H

#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <format>
#include <optional>
#include <algorithm>

// Platform-specific includes
#ifdef __has_include
  #if __has_include(<execinfo.h>)
    #define HAS_EXECINFO 1
    #include <execinfo.h>
  #endif
  #if __has_include(<cxxabi.h>)
    #define HAS_CXXABI 1
    #include <cxxabi.h>
  #endif
#endif

#ifdef _WIN32
  #include <windows.h>
  #include <dbghelp.h>
  #pragma comment(lib, "dbghelp.lib")
#endif

namespace fem::core::error {

/**
 * @brief Stack frame information
 */
struct StackFrame {
    size_t index = 0;                // Frame index in stack
    void* address = nullptr;          // Instruction pointer
    std::string module;               // Module/library name
    std::string function;             // Function name (demangled)
    std::string source_file;          // Source file if available
    size_t line_number = 0;           // Line number if available
    size_t offset = 0;                // Offset in function
    
    /**
     * @brief Format frame as string
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << std::format("#{:2d} ", index);
        
        if (!module.empty()) {
            oss << module << "!";
        }
        
        if (!function.empty()) {
            oss << function;
        } else if (address) {
            oss << std::format("0x{:016x}", reinterpret_cast<uintptr_t>(address));
        } else {
            oss << "<unknown>";
        }
        
        if (!source_file.empty()) {
            oss << " at " << source_file;
            if (line_number > 0) {
                oss << ":" << line_number;
            }
        } else if (offset > 0) {
            oss << std::format(" + 0x{:x}", offset);
        }
        
        return oss.str();
    }
    
    /**
     * @brief Check if frame is from system library
     */
    bool is_system() const {
        static const std::vector<std::string> system_prefixes = {
            "std::", "__", "_", "/usr/", "/lib/", "/System/", "ntdll", "kernel32"
        };
        
        for (const auto& prefix : system_prefixes) {
            if (function.starts_with(prefix) || module.starts_with(prefix)) {
                return true;
            }
        }
        return false;
    }
};

/**
 * @brief Stack trace capture and formatting
 */
class StackTrace {
public:
    /**
     * @brief Capture current stack trace
     */
    explicit StackTrace(size_t skip_frames = 1,
                       size_t max_frames = 64) {
        capture(skip_frames + 1, max_frames);
    }

    /**
     * @brief Default constructor (empty trace)
     */
    StackTrace() = default;

    /**
     * @brief Get captured frames
     */
    const std::vector<StackFrame>& frames() const { return frames_; }

    /**
     * @brief Check if trace is empty
     */
    bool empty() const { return frames_.empty(); }

    /**
     * @brief Get number of frames
     */
    size_t size() const { return frames_.size(); }

    /**
     * @brief Get specific frame
     */
    const StackFrame& operator[](size_t index) const {
        return frames_.at(index);
    }

    /**
     * @brief Format stack trace as string
     */
    std::string to_string(bool include_system_frames = false) const {
        if (frames_.empty()) {
            return "<no stack trace available>";
        }

        std::ostringstream oss;
        oss << "Stack trace:\n";
        
        for (const auto& frame : frames_) {
            if (!include_system_frames && frame.is_system()) {
                continue;
            }
            oss << "  " << frame.to_string() << "\n";
        }
        
        return oss.str();
    }

    /**
     * @brief Get compact trace (function names only)
     */
    std::string compact_trace(const std::string& separator = " <- ") const {
        std::ostringstream oss;
        bool first = true;
        
        for (const auto& frame : frames_) {
            if (frame.is_system()) {
                continue;
            }
            
            if (!first) {
                oss << separator;
            }
            
            if (!frame.function.empty()) {
                // Simplify function name
                auto simplified = simplify_function_name(frame.function);
                oss << simplified;
            } else {
                oss << "<unknown>";
            }
            
            first = false;
        }
        
        return oss.str();
    }

    /**
     * @brief Filter frames by module
     */
    StackTrace filter_module(const std::string& module_pattern) const {
        StackTrace filtered;
        
        for (const auto& frame : frames_) {
            if (frame.module.find(module_pattern) != std::string::npos) {
                filtered.frames_.push_back(frame);
            }
        }
        
        return filtered;
    }

    /**
     * @brief Filter frames by function
     */
    StackTrace filter_function(const std::string& function_pattern) const {
        StackTrace filtered;
        
        for (const auto& frame : frames_) {
            if (frame.function.find(function_pattern) != std::string::npos) {
                filtered.frames_.push_back(frame);
            }
        }
        
        return filtered;
    }

    /**
     * @brief Remove system frames
     */
    StackTrace without_system_frames() const {
        StackTrace filtered;
        
        for (const auto& frame : frames_) {
            if (!frame.is_system()) {
                filtered.frames_.push_back(frame);
            }
        }
        
        return filtered;
    }

    /**
     * @brief Get top N frames
     */
    StackTrace top(size_t n) const {
        StackTrace result;
        
        size_t count = std::min(n, frames_.size());
        result.frames_.assign(frames_.begin(), frames_.begin() + count);
        
        return result;
    }

    /**
     * @brief Find first frame matching pattern
     */
    std::optional<StackFrame> find_frame(const std::string& pattern) const {
        for (const auto& frame : frames_) {
            if (frame.function.find(pattern) != std::string::npos ||
                frame.module.find(pattern) != std::string::npos ||
                frame.source_file.find(pattern) != std::string::npos) {
                return frame;
            }
        }
        return std::nullopt;
    }

private:
    void capture(size_t skip_frames, size_t max_frames) {
#ifdef HAS_EXECINFO
        capture_execinfo(skip_frames, max_frames);
#elif defined(_WIN32)
        capture_windows(skip_frames, max_frames);
#else
        // No stack trace support on this platform
        frames_.clear();
#endif
    }

#ifdef HAS_EXECINFO
    void capture_execinfo(size_t skip_frames, size_t max_frames) {
        std::vector<void*> buffer(max_frames + skip_frames);
        
        int nframes = ::backtrace(buffer.data(), buffer.size());
        if (nframes <= static_cast<int>(skip_frames)) {
            return;
        }
        
        char** symbols = ::backtrace_symbols(buffer.data(), nframes);
        if (!symbols) {
            return;
        }
        
        frames_.reserve(nframes - skip_frames);
        
        for (int i = skip_frames; i < nframes; ++i) {
            StackFrame frame;
            frame.index = i - skip_frames;
            frame.address = buffer[i];
            
            // Parse the symbol string
            std::string symbol(symbols[i]);
            parse_symbol(symbol, frame);
            
            frames_.push_back(frame);
        }
        
        ::free(symbols);
    }
    
    void parse_symbol(const std::string& symbol, StackFrame& frame) {
        // Format varies by platform
        // Linux: module(function+offset) [address]
        // macOS: index module address function + offset
        
        size_t start = symbol.find('(');
        size_t plus = symbol.find('+', start);
        size_t end = symbol.find(')', plus);
        
        if (start != std::string::npos) {
            frame.module = symbol.substr(0, start);
            
            if (plus != std::string::npos && end != std::string::npos) {
                std::string mangled = symbol.substr(start + 1, plus - start - 1);
                frame.function = demangle(mangled);
                
                std::string offset_str = symbol.substr(plus + 1, end - plus - 1);
                if (!offset_str.empty() && offset_str[0] == '0') {
                    frame.offset = std::stoull(offset_str, nullptr, 16);
                }
            }
        } else {
            // Fallback: just store the whole symbol
            frame.function = symbol;
        }
    }
    
    std::string demangle(const std::string& mangled) {
#ifdef HAS_CXXABI
        int status = 0;
        char* demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
        
        if (status == 0 && demangled) {
            std::string result(demangled);
            ::free(demangled);
            return result;
        }
#endif
        return mangled;
    }
#endif // HAS_EXECINFO

#ifdef _WIN32
    void capture_windows(size_t skip_frames, size_t max_frames) {
        HANDLE process = GetCurrentProcess();
        SymInitialize(process, nullptr, TRUE);
        
        std::vector<void*> buffer(max_frames + skip_frames);
        WORD nframes = CaptureStackBackTrace(
            skip_frames, max_frames, buffer.data(), nullptr);
        
        frames_.reserve(nframes);
        
        SYMBOL_INFO* symbol = (SYMBOL_INFO*)calloc(
            sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
        symbol->MaxNameLen = 255;
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        
        for (WORD i = 0; i < nframes; ++i) {
            StackFrame frame;
            frame.index = i;
            frame.address = buffer[i];
            
            DWORD64 address = (DWORD64)buffer[i];
            
            if (SymFromAddr(process, address, nullptr, symbol)) {
                frame.function = symbol->Name;
                frame.offset = address - symbol->Address;
            }
            
            IMAGEHLP_MODULE64 module_info;
            module_info.SizeOfStruct = sizeof(IMAGEHLP_MODULE64);
            if (SymGetModuleInfo64(process, address, &module_info)) {
                frame.module = module_info.ModuleName;
            }
            
            DWORD displacement;
            IMAGEHLP_LINE64 line;
            line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
            if (SymGetLineFromAddr64(process, address, &displacement, &line)) {
                frame.source_file = line.FileName;
                frame.line_number = line.LineNumber;
            }
            
            frames_.push_back(frame);
        }
        
        free(symbol);
        SymCleanup(process);
    }
#endif // _WIN32

    static std::string simplify_function_name(const std::string& full_name) {
        // Remove template parameters
        std::string simplified = full_name;
        size_t pos = 0;
        int depth = 0;
        size_t start = std::string::npos;
        
        while (pos < simplified.length()) {
            if (simplified[pos] == '<') {
                if (depth == 0) {
                    start = pos;
                }
                depth++;
            } else if (simplified[pos] == '>') {
                depth--;
                if (depth == 0 && start != std::string::npos) {
                    simplified.erase(start, pos - start + 1);
                    pos = start;
                    start = std::string::npos;
                    continue;
                }
            }
            pos++;
        }
        
        // Remove parameters
        pos = simplified.find('(');
        if (pos != std::string::npos) {
            simplified = simplified.substr(0, pos);
        }
        
        // Get just the function name (remove namespace)
        pos = simplified.rfind("::");
        if (pos != std::string::npos) {
            simplified = simplified.substr(pos + 2);
        }
        
        return simplified;
    }

    std::vector<StackFrame> frames_;
};

/**
 * @brief Global stack trace utilities
 */
class StackTraceUtils {
public:
    /**
     * @brief Capture and format current stack trace
     */
    static std::string capture_string(size_t skip_frames = 1,
                                     size_t max_frames = 32,
                                     bool include_system = false) {
        StackTrace trace(skip_frames + 1, max_frames);
        return trace.to_string(include_system);
    }

    /**
     * @brief Get caller information
     */
    static std::string get_caller(size_t depth = 1) {
        StackTrace trace(depth + 1, depth + 2);
        if (!trace.empty()) {
            return trace[0].function;
        }
        return "<unknown>";
    }

    /**
     * @brief Check if currently in exception unwinding
     */
    static bool in_exception_handler() {
        return std::uncaught_exceptions() > 0;
    }
};

/**
 * @brief Macro helpers for stack traces
 */

// Capture current stack trace
#define FEM_STACK_TRACE() \
    fem::core::error::StackTrace(0)

// Get current caller
#define FEM_CALLER() \
    fem::core::error::StackTraceUtils::get_caller(0)

// Capture stack trace as string
#define FEM_STACK_STRING() \
    fem::core::error::StackTraceUtils::capture_string(0)

} // namespace fem::core::error

#endif // CORE_ERROR_STACK_TRACE_H