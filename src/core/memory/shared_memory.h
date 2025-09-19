#pragma once

#ifndef CORE_MEMORY_SHARED_MEMORY_H
#define CORE_MEMORY_SHARED_MEMORY_H

#include <cstddef>
#include <string>
#include <string_view>
#include <system_error>

#include <config/config.h>
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

class SharedMemory {
public:
    SharedMemory() = default;
    ~SharedMemory() { close(); }
    SharedMemory(const SharedMemory&) = delete;
    SharedMemory& operator=(const SharedMemory&) = delete;
    SharedMemory(SharedMemory&& o) noexcept { move_from(std::move(o)); }
    SharedMemory& operator=(SharedMemory&& o) noexcept { if (this != &o) { close(); move_from(std::move(o)); } return *this; }

    // Create or open a named shared memory segment. If create=true, ensures
    // at least 'size' bytes (truncate/extend as needed). Returns through ec.
    void open(std::string_view name, std::size_t size, bool create, std::error_code* ec = nullptr) noexcept;
    [[nodiscard]] fem::core::error::Result<void, fem::core::error::ErrorCode>
    open_result(std::string_view name, std::size_t size, bool create) noexcept {
        std::error_code ec; open(name, size, create, &ec);
        return ec ? fem::core::error::Err<fem::core::error::ErrorCode>(map_ec(ec, create))
                  : fem::core::error::Result<void, fem::core::error::ErrorCode>{};
    }
    void close() noexcept;
    static void unlink(std::string_view name, std::error_code* ec = nullptr) noexcept;
    [[nodiscard]] static fem::core::error::Result<void, fem::core::error::ErrorCode>
    unlink_result(std::string_view name) noexcept {
        std::error_code ec; unlink(name, &ec);
        return ec ? fem::core::error::Err<fem::core::error::ErrorCode>(map_ec(ec, /*create*/false))
                  : fem::core::error::Result<void, fem::core::error::ErrorCode>{};
    }

    [[nodiscard]] void* data() noexcept { return base_; }
    [[nodiscard]] const void* data() const noexcept { return base_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool valid() const noexcept { return base_ != nullptr; }

private:
    void* base_{nullptr};
    std::size_t size_{0};
#if defined(CORE_PLATFORM_WINDOWS)
    HANDLE mapping_{nullptr};
#else
    int fd_{-1};
    std::string name_{};
#endif

    void move_from(SharedMemory&& o) noexcept {
        base_ = o.base_; o.base_ = nullptr;
        size_ = o.size_; o.size_ = 0;
#if defined(CORE_PLATFORM_WINDOWS)
        mapping_ = o.mapping_; o.mapping_ = nullptr;
#else
        fd_ = o.fd_; o.fd_ = -1;
        name_ = std::move(o.name_);
#endif
    }
};

inline void SharedMemory::close() noexcept {
#if defined(CORE_PLATFORM_WINDOWS)
    if (base_) { ::UnmapViewOfFile(base_); base_ = nullptr; }
    if (mapping_) { ::CloseHandle(mapping_); mapping_ = nullptr; }
    size_ = 0;
#else
    if (base_) { ::munmap(base_, size_); base_ = nullptr; }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    size_ = 0; name_.clear();
#endif
}

inline void SharedMemory::open(std::string_view name, std::size_t size, bool create, std::error_code* ec) noexcept {
    std::error_code dummy; if (!ec) ec = &dummy; *ec = {};
    close();
#if defined(CORE_PLATFORM_WINDOWS)
    DWORD protect = PAGE_READWRITE;
    DWORD access = FILE_MAP_ALL_ACCESS;
    // Create or open mapping object backed by system paging file
    HANDLE hMap = ::CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, protect,
                                       0, static_cast<DWORD>(size), std::wstring(name.begin(), name.end()).c_str());
    if (!hMap) { *ec = std::error_code((int)::GetLastError(), std::system_category()); return; }
    mapping_ = hMap;
    void* base = ::MapViewOfFile(hMap, access, 0, 0, size);
    if (!base) { *ec = std::error_code((int)::GetLastError(), std::system_category()); close(); return; }
    base_ = base; size_ = size;
#else
    int oflag = create ? (O_RDWR | O_CREAT) : O_RDWR;
    std::string n(name);
    if (n.empty() || n[0] != '/') n = "/" + n; // POSIX requires leading '/'
    int fd = ::shm_open(n.c_str(), oflag, 0666);
    if (fd < 0) { *ec = std::error_code(errno, std::generic_category()); return; }
    fd_ = fd; name_ = n;
    if (create) {
        if (::ftruncate(fd, static_cast<off_t>(size)) != 0) { *ec = std::error_code(errno, std::generic_category()); close(); return; }
    } else {
        struct ::stat st{}; if (::fstat(fd, &st) != 0) { *ec = std::error_code(errno, std::generic_category()); close(); return; }
        if (static_cast<std::size_t>(st.st_size) < size) { *ec = std::make_error_code(std::errc::file_too_large); close(); return; }
    }
    void* base = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { *ec = std::error_code(errno, std::generic_category()); close(); return; }
    base_ = base; size_ = size;
#endif
}

inline void SharedMemory::unlink(std::string_view name, std::error_code* ec) noexcept {
    std::error_code dummy; if (!ec) ec = &dummy; *ec = {};
#if defined(CORE_PLATFORM_WINDOWS)
    // No global unlink needed; mapping is released on last handle close.
    (void)name;
#else
    std::string n(name);
    if (n.empty() || n[0] != '/') n = "/" + n;
    if (::shm_unlink(n.c_str()) != 0) { *ec = std::error_code(errno, std::generic_category()); }
#endif
}

// Map std::error_code to core ErrorCode for shared memory ops
inline fem::core::error::ErrorCode map_ec(const std::error_code& ec, bool create) noexcept {
#if defined(CORE_PLATFORM_WINDOWS)
    // Windows named mappings via paging file: treat failures as system or access errors
    switch (ec.value()) {
        case ERROR_ACCESS_DENIED: return fem::core::error::ErrorCode::FileAccessDenied;
        case ERROR_NOT_ENOUGH_MEMORY: case ERROR_OUTOFMEMORY: return fem::core::error::ErrorCode::OutOfMemory;
        default: return fem::core::error::ErrorCode::SystemError;
    }
#else
    switch (ec.value()) {
        case ENOENT: return create ? fem::core::error::ErrorCode::IoError
                                   : fem::core::error::ErrorCode::ResourceNotFound;
        case EACCES: return fem::core::error::ErrorCode::FileAccessDenied;
        case EEXIST: return fem::core::error::ErrorCode::FileAlreadyExists;
        case ENOMEM: return fem::core::error::ErrorCode::OutOfMemory;
        default: return fem::core::error::ErrorCode::SystemError;
    }
#endif
}

} // namespace fem::core::memory

#endif // CORE_MEMORY_SHARED_MEMORY_H
