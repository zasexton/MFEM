#pragma once

#ifndef CORE_MEMORY_MEMORY_MAPPED_H
#define CORE_MEMORY_MEMORY_MAPPED_H

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <system_error>
#include <string>
#include <type_traits>

#include <config/config.h>
#include <config/debug.h>
#include <core/error/result.h>
#include <core/error/error_code.h>

#if defined(CORE_PLATFORM_WINDOWS)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#  include <errno.h>
#endif

namespace fem::core::memory {

// Map std::error_code or platform error to core ErrorCode
inline fem::core::error::ErrorCode map_ec(const std::error_code& ec) noexcept {
#if !defined(CORE_PLATFORM_WINDOWS)
    switch (ec.value()) {
        case ENOENT: return fem::core::error::ErrorCode::FileNotFound;
        case EACCES: return fem::core::error::ErrorCode::FileAccessDenied;
        case EEXIST: return fem::core::error::ErrorCode::FileAlreadyExists;
        case ENOTDIR: case EINVAL: case EFAULT: case ENAMETOOLONG:
            return fem::core::error::ErrorCode::InvalidPath;
        case EIO: return fem::core::error::ErrorCode::IoError;
        case ENOMEM: return fem::core::error::ErrorCode::OutOfMemory;
        default: return fem::core::error::ErrorCode::SystemError;
    }
#else
    switch (ec.value()) {
        case ERROR_FILE_NOT_FOUND: return fem::core::error::ErrorCode::FileNotFound;
        case ERROR_PATH_NOT_FOUND: return fem::core::error::ErrorCode::InvalidPath;
        case ERROR_ACCESS_DENIED: return fem::core::error::ErrorCode::FileAccessDenied;
        case ERROR_ALREADY_EXISTS: case ERROR_FILE_EXISTS: return fem::core::error::ErrorCode::FileAlreadyExists;
        case ERROR_NOT_ENOUGH_MEMORY: case ERROR_OUTOFMEMORY: return fem::core::error::ErrorCode::OutOfMemory;
        default: return fem::core::error::ErrorCode::SystemError;
    }
#endif
}

class MemoryMappedFile;

class MemoryMappedView {
public:
    MemoryMappedView() = default;
    MemoryMappedView(const MemoryMappedView&) = default;
    MemoryMappedView& operator=(const MemoryMappedView&) = default;

    [[nodiscard]] void* data() noexcept { return base_; }
    [[nodiscard]] const void* data() const noexcept { return base_; }
    template<class T = std::byte>
    [[nodiscard]] T* data_as() noexcept { return static_cast<T*>(base_); }
    template<class T = std::byte>
    [[nodiscard]] const T* data_as() const noexcept { return static_cast<const T*>(base_); }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool writable() const noexcept { return writable_; }

    // Flush the view (offset within the view)
    void flush(std::size_t offset = 0, std::size_t length = 0, std::error_code* ec = nullptr) noexcept;

private:
    friend class MemoryMappedFile;
    void* base_{nullptr};
    std::size_t size_{0};
    MemoryMappedFile* parent_{nullptr};
    std::size_t parent_offset_{0};
    bool writable_{false};
};

class MemoryMappedFile {
public:
    enum class Mode { ReadOnly, ReadWrite, CopyOnWrite };

    MemoryMappedFile() = default;
    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;
    MemoryMappedFile(MemoryMappedFile&& other) noexcept { move_from(std::move(other)); }
    MemoryMappedFile& operator=(MemoryMappedFile&& other) noexcept {
        if (this != &other) { close(); move_from(std::move(other)); }
        return *this;
    }
    ~MemoryMappedFile() { close(); }

    // Open existing file (ReadOnly/CopyOnWrite) or existing RW file (ReadWrite)
    MemoryMappedFile(const std::filesystem::path& path, Mode mode) { open(path, mode); }

    // Open and ensure file at least 'length' bytes (ReadWrite)
    MemoryMappedFile(const std::filesystem::path& path, Mode mode, std::size_t length) { open(path, mode, length); }

    void open(const std::filesystem::path& path, Mode mode) { std::error_code ec; open(path, mode, 0, &ec); throw_if(ec); }
    void open(const std::filesystem::path& path, Mode mode, std::size_t length) { std::error_code ec; open(path, mode, length, &ec); throw_if(ec); }
    void open(const std::filesystem::path& path, Mode mode, std::size_t length, std::error_code* ec) noexcept;

    // Result-based convenience wrappers using core/error
    [[nodiscard]] fem::core::error::Result<void, fem::core::error::ErrorCode>
    open_result(const std::filesystem::path& path, Mode mode) noexcept {
        std::error_code ec; open(path, mode, 0, &ec);
        return ec ? fem::core::error::Err<fem::core::error::ErrorCode>(map_ec(ec))
                  : fem::core::error::Result<void, fem::core::error::ErrorCode>{};
    }

    [[nodiscard]] fem::core::error::Result<void, fem::core::error::ErrorCode>
    open_result(const std::filesystem::path& path, Mode mode, std::size_t length) noexcept {
        std::error_code ec; open(path, mode, length, &ec);
        return ec ? fem::core::error::Err<fem::core::error::ErrorCode>(map_ec(ec))
                  : fem::core::error::Result<void, fem::core::error::ErrorCode>{};
    }

    void close() noexcept;

    [[nodiscard]] void* data() noexcept { return base_; }
    [[nodiscard]] const void* data() const noexcept { return base_; }
    template<class T = std::byte>
    [[nodiscard]] T* data_as() noexcept { return static_cast<T*>(base_); }
    template<class T = std::byte>
    [[nodiscard]] const T* data_as() const noexcept { return static_cast<const T*>(base_); }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool writable() const noexcept { return writable_; }
    [[nodiscard]] Mode mode() const noexcept { return mode_; }

    // Flush entire mapping or specified range
    void flush(std::size_t offset = 0, std::size_t length = 0, std::error_code* ec = nullptr) noexcept;
    [[nodiscard]] fem::core::error::Result<void, fem::core::error::ErrorCode>
    flush_result(std::size_t offset = 0, std::size_t length = 0) const noexcept {
        std::error_code ec; const_cast<MemoryMappedFile*>(this)->flush(offset, length, &ec);
        return ec ? fem::core::error::Err<fem::core::error::ErrorCode>(map_ec(ec))
                  : fem::core::error::Result<void, fem::core::error::ErrorCode>{};
    }

    // Advisory hints (best effort)
    void advise_sequential(std::error_code* ec = nullptr) noexcept;
    void advise_random(std::error_code* ec = nullptr) noexcept;
    void lock_in_memory(std::error_code* ec = nullptr) noexcept;
    [[nodiscard]] fem::core::error::Result<void, fem::core::error::ErrorCode> advise_sequential_result() const noexcept {
        std::error_code ec; const_cast<MemoryMappedFile*>(this)->advise_sequential(&ec);
        return ec ? fem::core::error::Err<fem::core::error::ErrorCode>(map_ec(ec))
                  : fem::core::error::Result<void, fem::core::error::ErrorCode>{};
    }
    [[nodiscard]] fem::core::error::Result<void, fem::core::error::ErrorCode> advise_random_result() const noexcept {
        std::error_code ec; const_cast<MemoryMappedFile*>(this)->advise_random(&ec);
        return ec ? fem::core::error::Err<fem::core::error::ErrorCode>(map_ec(ec))
                  : fem::core::error::Result<void, fem::core::error::ErrorCode>{};
    }
    [[nodiscard]] fem::core::error::Result<void, fem::core::error::ErrorCode> lock_in_memory_result() const noexcept {
        std::error_code ec; const_cast<MemoryMappedFile*>(this)->lock_in_memory(&ec);
        return ec ? fem::core::error::Err<fem::core::error::ErrorCode>(map_ec(ec))
                  : fem::core::error::Result<void, fem::core::error::ErrorCode>{};
    }

    // Create a logical sub-view (no additional OS mapping)
    [[nodiscard]] MemoryMappedView create_view(std::size_t offset, std::size_t length) noexcept {
        MemoryMappedView v;
        if (!base_ || offset > size_ || length == 0 || offset + length > size_) return v;
        v.base_ = static_cast<std::byte*>(base_) + offset;
        v.size_ = length;
        v.parent_ = this;
        v.parent_offset_ = offset;
        v.writable_ = writable_;
        return v;
    }

private:
    static void throw_if(const std::error_code& ec) {
#if CORE_ENABLE_EXCEPTIONS
        if (ec) throw std::system_error(ec);
#else
        CORE_UNUSED(ec);
#endif
    }

    void move_from(MemoryMappedFile&& o) noexcept {
        base_ = o.base_; o.base_ = nullptr;
        size_ = o.size_; o.size_ = 0;
        writable_ = o.writable_; o.writable_ = false;
        mode_ = o.mode_;
#if defined(CORE_PLATFORM_WINDOWS)
        file_ = o.file_; o.file_ = (void*)-1;
        mapping_ = o.mapping_; o.mapping_ = nullptr;
#else
        fd_ = o.fd_; o.fd_ = -1;
#endif
    }

private:
    void* base_{nullptr};
    std::size_t size_{0};
    bool writable_{false};
    Mode mode_{Mode::ReadOnly};

#if defined(CORE_PLATFORM_WINDOWS)
    void* file_{(void*)-1};
    void* mapping_{nullptr};
#else
    int fd_{-1};
#endif
};

// ============================= Implementation ==============================

inline void MemoryMappedView::flush(std::size_t offset, std::size_t length, std::error_code* ec) noexcept {
    if (!parent_) { if (ec) *ec = {}; return; }
    parent_->flush(parent_offset_ + offset, length, ec);
}

inline void MemoryMappedFile::close() noexcept {
#if defined(CORE_PLATFORM_WINDOWS)
    if (base_) {
        // Flush best-effort; ignore errors
        // ::FlushViewOfFile(base_, 0);
        ::UnmapViewOfFile(base_);
        base_ = nullptr;
    }
    if (mapping_) { ::CloseHandle(mapping_); mapping_ = nullptr; }
    if (file_ && file_ != (void*)-1) { ::CloseHandle(file_); file_ = (void*)-1; }
    size_ = 0; writable_ = false;
#else
    if (base_) {
        // Flush best-effort; ignore errors
        // ::msync(base_, size_, MS_SYNC);
        ::munmap(base_, size_);
        base_ = nullptr;
    }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    size_ = 0; writable_ = false;
#endif
}

inline void MemoryMappedFile::open(const std::filesystem::path& path, Mode mode, std::size_t length, std::error_code* ec) noexcept {
    close();
    std::error_code dummy;
    if (!ec) ec = &dummy;
    *ec = {};
    mode_ = mode;

#if defined(CORE_PLATFORM_WINDOWS)
    // Windows implementation
    DWORD access = 0, share = FILE_SHARE_READ;
    DWORD creation = OPEN_EXISTING;
    DWORD protect = 0, map_access = 0;

    switch (mode) {
        case Mode::ReadOnly:
            access = GENERIC_READ; creation = OPEN_EXISTING; break;
        case Mode::ReadWrite:
            access = GENERIC_READ | GENERIC_WRITE; creation = OPEN_ALWAYS; break;
        case Mode::CopyOnWrite:
            access = GENERIC_READ; creation = OPEN_EXISTING; break;
    }

    HANDLE hFile = ::CreateFileW(path.wstring().c_str(), access, share, nullptr, creation, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) { *ec = std::error_code((int)::GetLastError(), std::system_category()); return; }
    file_ = hFile;

    LARGE_INTEGER fs{};
    if (!::GetFileSizeEx(hFile, &fs)) { *ec = std::error_code((int)::GetLastError(), std::system_category()); close(); return; }
    std::size_t fsize = static_cast<std::size_t>(fs.QuadPart);
    if (mode == Mode::ReadWrite && length > fsize) {
        // Extend file
        LARGE_INTEGER li; li.QuadPart = static_cast<LONGLONG>(length);
        if (!::SetFilePointerEx(hFile, li, nullptr, FILE_BEGIN) || !::SetEndOfFile(hFile)) {
            *ec = std::error_code((int)::GetLastError(), std::system_category()); close(); return; }
        fsize = length;
    }
    size_ = fsize;

    switch (mode) {
        case Mode::ReadOnly:   protect = PAGE_READONLY;               map_access = FILE_MAP_READ; break;
        case Mode::ReadWrite:  protect = PAGE_READWRITE;              map_access = FILE_MAP_READ | FILE_MAP_WRITE; writable_ = true; break;
        case Mode::CopyOnWrite:protect = PAGE_WRITECOPY;              map_access = FILE_MAP_COPY; writable_ = true; break;
    }

    HANDLE hMap = ::CreateFileMappingW(hFile, nullptr, protect, 0, 0, nullptr);
    if (!hMap) { *ec = std::error_code((int)::GetLastError(), std::system_category()); close(); return; }
    mapping_ = hMap;
    void* base = ::MapViewOfFile(hMap, map_access, 0, 0, 0);
    if (!base) { *ec = std::error_code((int)::GetLastError(), std::system_category()); close(); return; }
    base_ = base;

#else
    // POSIX implementation
    int flags = 0, prot = 0, map_flags = 0;
    switch (mode) {
        case Mode::ReadOnly:   flags = O_RDONLY; break;
        case Mode::ReadWrite:  flags = O_RDWR | O_CREAT; break;
        case Mode::CopyOnWrite:flags = O_RDONLY; break;
    }
    int fd = ::open(path.c_str(), flags, 0644);
    if (fd < 0) { *ec = std::error_code(errno, std::generic_category()); return; }
    fd_ = fd;

    // Determine/ensure size
    struct ::stat st{};
    if (::fstat(fd, &st) != 0) { *ec = std::error_code(errno, std::generic_category()); close(); return; }
    std::size_t fsize = static_cast<std::size_t>(st.st_size);
    if (mode == Mode::ReadWrite && length > fsize) {
        if (::ftruncate(fd, static_cast<off_t>(length)) != 0) { *ec = std::error_code(errno, std::generic_category()); close(); return; }
        fsize = length;
    }
    size_ = fsize;

    switch (mode) {
        case Mode::ReadOnly:    prot = PROT_READ; map_flags = MAP_SHARED; writable_ = false; break;
        case Mode::ReadWrite:   prot = PROT_READ | PROT_WRITE; map_flags = MAP_SHARED; writable_ = true; break;
        case Mode::CopyOnWrite: prot = PROT_READ | PROT_WRITE; map_flags = MAP_PRIVATE; writable_ = true; break; // writes not persisted
    }

    void* base = ::mmap(nullptr, size_, prot, map_flags, fd, 0);
    if (base == MAP_FAILED) {
        *ec = std::error_code(errno, std::generic_category()); close(); return; }
    base_ = base;
#endif
}

inline void MemoryMappedFile::flush(std::size_t offset, std::size_t length, std::error_code* ec) noexcept {
    if (!base_) { if (ec) *ec = {}; return; }
    if (length == 0 || offset + length > size_) length = size_ - offset;
#if defined(CORE_PLATFORM_WINDOWS)
    BOOL ok = ::FlushViewOfFile(static_cast<char*>(base_) + offset, length);
    if (!ok && ec) *ec = std::error_code((int)::GetLastError(), std::system_category());
#else
    int r = ::msync(static_cast<char*>(base_) + offset, length, MS_SYNC);
    if (r != 0 && ec) *ec = std::error_code(errno, std::generic_category());
#endif
}

inline void MemoryMappedFile::advise_sequential(std::error_code* ec) noexcept {
#if defined(CORE_PLATFORM_WINDOWS)
    CORE_UNUSED(ec);
#else
    if (!base_) { if (ec) *ec = {}; return; }
#ifdef MADV_SEQUENTIAL
    int r = ::madvise(base_, size_, MADV_SEQUENTIAL);
    if (r != 0 && ec) *ec = std::error_code(errno, std::generic_category());
#else
    CORE_UNUSED(ec);
#endif
#endif
}

inline void MemoryMappedFile::advise_random(std::error_code* ec) noexcept {
#if defined(CORE_PLATFORM_WINDOWS)
    CORE_UNUSED(ec);
#else
    if (!base_) { if (ec) *ec = {}; return; }
#ifdef MADV_RANDOM
    int r = ::madvise(base_, size_, MADV_RANDOM);
    if (r != 0 && ec) *ec = std::error_code(errno, std::generic_category());
#else
    CORE_UNUSED(ec);
#endif
#endif
}

inline void MemoryMappedFile::lock_in_memory(std::error_code* ec) noexcept {
#if defined(CORE_PLATFORM_WINDOWS)
    CORE_UNUSED(ec);
#else
    if (!base_) { if (ec) *ec = {}; return; }
#ifdef MLOCK_ONFAULT
    int r = ::mlock2(base_, size_, MLOCK_ONFAULT);
    if (r != 0 && ec) *ec = std::error_code(errno, std::generic_category());
#else
    int r = ::mlock(base_, size_);
    if (r != 0 && ec) *ec = std::error_code(errno, std::generic_category());
#endif
#endif
}

} // namespace fem::core::memory

#endif // CORE_MEMORY_MEMORY_MAPPED_H
