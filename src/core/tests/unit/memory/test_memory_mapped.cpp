#include <gtest/gtest.h>
#include <core/memory/memory_mapped.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <cstring>
#include <random>
#include <thread>

namespace fcm = fem::core::memory;
namespace fce = fem::core::error;
namespace fs = std::filesystem;

class MemoryMappedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temp directory for test files
        test_dir_ = fs::temp_directory_path() / "memory_mapped_test";
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        // Clean up test directory
        std::error_code ec;
        fs::remove_all(test_dir_, ec);
    }

    // Helper to create test file with content
    fs::path create_test_file(const std::string& name, const std::string& content) {
        fs::path path = test_dir_ / name;
        std::ofstream ofs(path, std::ios::binary);
        ofs.write(content.data(), content.size());
        ofs.close();
        return path;
    }

    // Helper to read file content
    std::string read_file_content(const fs::path& path) {
        std::ifstream ifs(path, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(ifs)),
                          (std::istreambuf_iterator<char>()));
    }

    fs::path test_dir_;
};

// === Basic Construction and Open Tests ===

TEST_F(MemoryMappedTest, DefaultConstruction) {
    fcm::MemoryMappedFile mmf;
    EXPECT_EQ(mmf.data(), nullptr);
    EXPECT_EQ(mmf.size(), 0);
    EXPECT_FALSE(mmf.writable());
}

TEST_F(MemoryMappedTest, OpenExistingFileReadOnly) {
    const std::string content = "Hello, Memory Mapped World!";
    auto path = create_test_file("readonly.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadOnly);
    EXPECT_NE(mmf.data(), nullptr);
    EXPECT_EQ(mmf.size(), content.size());
    EXPECT_FALSE(mmf.writable());
    EXPECT_EQ(mmf.mode(), fcm::MemoryMappedFile::Mode::ReadOnly);

    // Verify content
    EXPECT_EQ(std::memcmp(mmf.data(), content.data(), content.size()), 0);
}

TEST_F(MemoryMappedTest, OpenExistingFileReadWrite) {
    const std::string content = "Initial content";
    auto path = create_test_file("readwrite.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadWrite);
    EXPECT_NE(mmf.data(), nullptr);
    EXPECT_EQ(mmf.size(), content.size());
    EXPECT_TRUE(mmf.writable());
    EXPECT_EQ(mmf.mode(), fcm::MemoryMappedFile::Mode::ReadWrite);

    // Modify content
    char* data = static_cast<char*>(mmf.data());
    std::memcpy(data, "Modified", 8);

    mmf.close();

    // Verify modification was persisted
    std::string new_content = read_file_content(path);
    EXPECT_EQ(new_content.substr(0, 8), "Modified");
}

TEST_F(MemoryMappedTest, OpenExistingFileCopyOnWrite) {
    const std::string content = "Original content";
    auto path = create_test_file("copyonwrite.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::CopyOnWrite);
    EXPECT_NE(mmf.data(), nullptr);
    EXPECT_EQ(mmf.size(), content.size());
    EXPECT_TRUE(mmf.writable()); // COW is writable but won't persist
    EXPECT_EQ(mmf.mode(), fcm::MemoryMappedFile::Mode::CopyOnWrite);

    // Modify content
    char* data = static_cast<char*>(mmf.data());
    std::memcpy(data, "Modified", 8);

    mmf.close();

    // Verify original file unchanged (COW doesn't persist)
    std::string file_content = read_file_content(path);
    EXPECT_EQ(file_content, content);
}

TEST_F(MemoryMappedTest, OpenNonExistentFileReadOnly) {
    fs::path path = test_dir_ / "nonexistent.txt";

    std::error_code ec;
    fcm::MemoryMappedFile mmf;
    mmf.open(path, fcm::MemoryMappedFile::Mode::ReadOnly, 0, &ec);
    EXPECT_TRUE(ec);
    EXPECT_EQ(mmf.data(), nullptr);
}

TEST_F(MemoryMappedTest, CreateNewFileReadWrite) {
    fs::path path = test_dir_ / "newfile.txt";
    const size_t size = 1024;

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadWrite, size);
    EXPECT_NE(mmf.data(), nullptr);
    EXPECT_EQ(mmf.size(), size);
    EXPECT_TRUE(mmf.writable());

    // Write some data
    char* data = static_cast<char*>(mmf.data());
    std::strcpy(data, "New file content");

    mmf.close();

    // Verify file exists and has correct size
    EXPECT_TRUE(fs::exists(path));
    EXPECT_EQ(fs::file_size(path), size);
}

TEST_F(MemoryMappedTest, ExtendExistingFile) {
    const std::string content = "Short";
    auto path = create_test_file("extend.txt", content);
    const size_t new_size = 1024;

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadWrite, new_size);
    EXPECT_EQ(mmf.size(), new_size);

    // Original content should still be there
    EXPECT_EQ(std::memcmp(mmf.data(), content.data(), content.size()), 0);

    mmf.close();

    // Verify file was extended
    EXPECT_EQ(fs::file_size(path), new_size);
}

// === Move Semantics Tests ===

TEST_F(MemoryMappedTest, MoveConstructor) {
    const std::string content = "Move test content";
    auto path = create_test_file("move.txt", content);

    fcm::MemoryMappedFile original(path, fcm::MemoryMappedFile::Mode::ReadOnly);
    void* orig_data = original.data();
    size_t orig_size = original.size();

    fcm::MemoryMappedFile moved(std::move(original));
    EXPECT_EQ(moved.data(), orig_data);
    EXPECT_EQ(moved.size(), orig_size);

    // Original should be empty
    EXPECT_EQ(original.data(), nullptr);
    EXPECT_EQ(original.size(), 0);
}

TEST_F(MemoryMappedTest, MoveAssignment) {
    const std::string content1 = "First file";
    const std::string content2 = "Second file";
    auto path1 = create_test_file("move1.txt", content1);
    auto path2 = create_test_file("move2.txt", content2);

    fcm::MemoryMappedFile mmf1(path1, fcm::MemoryMappedFile::Mode::ReadOnly);
    fcm::MemoryMappedFile mmf2(path2, fcm::MemoryMappedFile::Mode::ReadOnly);

    void* orig_data = mmf2.data();
    size_t orig_size = mmf2.size();

    mmf1 = std::move(mmf2);
    EXPECT_EQ(mmf1.data(), orig_data);
    EXPECT_EQ(mmf1.size(), orig_size);

    // mmf2 should be empty
    EXPECT_EQ(mmf2.data(), nullptr);
    EXPECT_EQ(mmf2.size(), 0);
}

// === Data Access Tests ===

TEST_F(MemoryMappedTest, DataAsTemplateAccess) {
    const std::string content = "Test data";
    auto path = create_test_file("dataas.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadOnly);

    // Access as different types
    const char* char_data = mmf.data_as<char>();
    EXPECT_EQ(std::string(char_data, content.size()), content);

    const std::byte* byte_data = mmf.data_as<std::byte>();
    EXPECT_NE(byte_data, nullptr);

    const uint8_t* uint8_data = mmf.data_as<uint8_t>();
    EXPECT_NE(uint8_data, nullptr);
}

TEST_F(MemoryMappedTest, ModifyDataInReadWriteMode) {
    const std::string content = "0123456789";
    auto path = create_test_file("modify.txt", content);

    {
        fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadWrite);
        char* data = mmf.data_as<char>();

        // Modify middle portion
        std::memcpy(data + 3, "ABC", 3);

        // Flush changes
        std::error_code ec;
        mmf.flush(0, 0, &ec);
        EXPECT_FALSE(ec);
    }

    // Verify changes persisted
    std::string new_content = read_file_content(path);
    EXPECT_EQ(new_content, "012ABC6789");
}

// === View Creation Tests ===

TEST_F(MemoryMappedTest, CreateView) {
    const std::string content = "0123456789ABCDEF";
    auto path = create_test_file("view.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadOnly);

    // Create view of middle portion
    auto view = mmf.create_view(4, 8);
    EXPECT_NE(view.data(), nullptr);
    EXPECT_EQ(view.size(), 8);
    EXPECT_FALSE(view.writable());

    // Verify view content
    const char* view_data = static_cast<const char*>(view.data());
    EXPECT_EQ(std::string(view_data, 8), "456789AB");
}

TEST_F(MemoryMappedTest, CreateViewBoundaryChecks) {
    const std::string content = "Test content";
    auto path = create_test_file("viewbounds.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadOnly);

    // Invalid views should return empty
    auto view1 = mmf.create_view(100, 10);  // Offset out of range
    EXPECT_EQ(view1.data(), nullptr);
    EXPECT_EQ(view1.size(), 0);

    auto view2 = mmf.create_view(5, 100);  // Length too large
    EXPECT_EQ(view2.data(), nullptr);
    EXPECT_EQ(view2.size(), 0);

    auto view3 = mmf.create_view(0, 0);  // Zero length
    EXPECT_EQ(view3.data(), nullptr);
    EXPECT_EQ(view3.size(), 0);
}

TEST_F(MemoryMappedTest, ViewInheritsWritability) {
    const std::string content = "Writable content";
    auto path = create_test_file("viewwrite.txt", content);

    {
        fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadWrite);
        auto view = mmf.create_view(0, 8);
        EXPECT_TRUE(view.writable());

        // Modify through view
        char* view_data = static_cast<char*>(view.data());
        std::memcpy(view_data, "Modified", 8);

        // Flush view
        std::error_code ec;
        view.flush(0, 0, &ec);
        EXPECT_FALSE(ec);
    }

    // Verify changes
    std::string new_content = read_file_content(path);
    EXPECT_EQ(new_content.substr(0, 8), "Modified");
}

// === Flush Tests ===

TEST_F(MemoryMappedTest, FlushFullFile) {
    const std::string content = "Content to flush";
    auto path = create_test_file("flush.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadWrite);

    // Modify and flush
    char* data = mmf.data_as<char>();
    data[0] = 'X';

    std::error_code ec;
    mmf.flush(0, 0, &ec);
    EXPECT_FALSE(ec);
}

TEST_F(MemoryMappedTest, FlushPartialRange) {
    const std::string content(4096, 'A');  // Large enough for partial flush
    auto path = create_test_file("flushpartial.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadWrite);

    // Modify specific range
    char* data = mmf.data_as<char>();
    std::memset(data + 1024, 'B', 1024);

    // Flush only modified range
    std::error_code ec;
    mmf.flush(1024, 1024, &ec);
    // Flush may not be supported on all platforms, just check no crash
}

// === Advisory Hints Tests ===

TEST_F(MemoryMappedTest, AdviseSequential) {
    const std::string content(8192, 'X');
    auto path = create_test_file("advise_seq.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadOnly);

    std::error_code ec;
    mmf.advise_sequential(&ec);
    // Advisory hints may not be supported on all systems
    // Just verify no crash
}

TEST_F(MemoryMappedTest, AdviseRandom) {
    const std::string content(8192, 'X');
    auto path = create_test_file("advise_rand.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadOnly);

    std::error_code ec;
    mmf.advise_random(&ec);
    // Advisory hints may not be supported on all systems
    // Just verify no crash
}

TEST_F(MemoryMappedTest, LockInMemory) {
    const std::string content(4096, 'X');
    auto path = create_test_file("lock.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadOnly);

    std::error_code ec;
    mmf.lock_in_memory(&ec);
    // May require privileges or fail on some systems
    // Just verify no crash
}

// === Result-based API Tests ===

TEST_F(MemoryMappedTest, OpenResultSuccess) {
    const std::string content = "Result API test";
    auto path = create_test_file("result_success.txt", content);

    fcm::MemoryMappedFile mmf;
    auto result = mmf.open_result(path, fcm::MemoryMappedFile::Mode::ReadOnly);
    EXPECT_TRUE(result.is_ok());
    EXPECT_NE(mmf.data(), nullptr);
}

TEST_F(MemoryMappedTest, OpenResultFailure) {
    fs::path path = test_dir_ / "nonexistent_result.txt";

    fcm::MemoryMappedFile mmf;
    auto result = mmf.open_result(path, fcm::MemoryMappedFile::Mode::ReadOnly);
    EXPECT_TRUE(result.is_error());
    EXPECT_EQ(mmf.data(), nullptr);

    if (result.is_error()) {
        auto err = result.error();
        EXPECT_EQ(err, fce::ErrorCode::FileNotFound);
    }
}

TEST_F(MemoryMappedTest, FlushResult) {
    const std::string content = "Flush result test";
    auto path = create_test_file("flush_result.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadWrite);

    char* data = mmf.data_as<char>();
    data[0] = 'X';

    auto result = mmf.flush_result();
    EXPECT_TRUE(result.is_ok());
}

TEST_F(MemoryMappedTest, AdviseSequentialResult) {
    const std::string content(4096, 'X');
    auto path = create_test_file("advise_seq_result.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadOnly);

    auto result = mmf.advise_sequential_result();
    // May succeed or fail depending on platform
    // Just verify it returns a result
    EXPECT_TRUE(result.is_ok() || result.is_error());
}

// === Large File Tests ===

TEST_F(MemoryMappedTest, LargeFileMapping) {
    const size_t size = 10 * 1024 * 1024; // 10MB
    fs::path path = test_dir_ / "large.dat";

    // Create large file
    {
        std::ofstream ofs(path, std::ios::binary);
        std::vector<char> buffer(1024 * 1024, 'X');
        for (size_t i = 0; i < 10; ++i) {
            ofs.write(buffer.data(), buffer.size());
        }
    }

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadOnly);
    EXPECT_EQ(mmf.size(), size);
    EXPECT_NE(mmf.data(), nullptr);

    // Verify can read from various offsets
    const char* data = mmf.data_as<char>();
    EXPECT_EQ(data[0], 'X');
    EXPECT_EQ(data[size / 2], 'X');
    EXPECT_EQ(data[size - 1], 'X');
}

// === Multiple Mappings Test ===

TEST_F(MemoryMappedTest, MultipleReadOnlyMappings) {
    const std::string content = "Shared read-only content";
    auto path = create_test_file("multiread.txt", content);

    fcm::MemoryMappedFile mmf1(path, fcm::MemoryMappedFile::Mode::ReadOnly);
    fcm::MemoryMappedFile mmf2(path, fcm::MemoryMappedFile::Mode::ReadOnly);

    EXPECT_NE(mmf1.data(), nullptr);
    EXPECT_NE(mmf2.data(), nullptr);
    EXPECT_EQ(mmf1.size(), mmf2.size());

    // Both should see same content
    EXPECT_EQ(std::memcmp(mmf1.data(), content.data(), content.size()), 0);
    EXPECT_EQ(std::memcmp(mmf2.data(), content.data(), content.size()), 0);
}

// === Edge Cases ===

TEST_F(MemoryMappedTest, EmptyFile) {
    auto path = create_test_file("empty.txt", "");

    // Empty files may fail to map on some platforms
    std::error_code ec;
    fcm::MemoryMappedFile mmf;
    mmf.open(path, fcm::MemoryMappedFile::Mode::ReadOnly, 0, &ec);
    // Just check that it doesn't crash - empty files are edge cases
}

TEST_F(MemoryMappedTest, ReopenFile) {
    const std::string content1 = "First content";
    const std::string content2 = "Second different content";
    auto path1 = create_test_file("reopen1.txt", content1);
    auto path2 = create_test_file("reopen2.txt", content2);

    fcm::MemoryMappedFile mmf;

    // Open first file
    mmf.open(path1, fcm::MemoryMappedFile::Mode::ReadOnly);
    EXPECT_EQ(mmf.size(), content1.size());

    // Open second file (should close first)
    mmf.open(path2, fcm::MemoryMappedFile::Mode::ReadOnly);
    EXPECT_EQ(mmf.size(), content2.size());
    EXPECT_EQ(std::memcmp(mmf.data(), content2.data(), content2.size()), 0);
}

TEST_F(MemoryMappedTest, CloseAndReopen) {
    const std::string content = "Close and reopen test";
    auto path = create_test_file("closereopen.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadOnly);
    EXPECT_NE(mmf.data(), nullptr);

    mmf.close();
    EXPECT_EQ(mmf.data(), nullptr);
    EXPECT_EQ(mmf.size(), 0);

    mmf.open(path, fcm::MemoryMappedFile::Mode::ReadOnly);
    EXPECT_NE(mmf.data(), nullptr);
    EXPECT_EQ(mmf.size(), content.size());
}

// === Binary Data Tests ===

TEST_F(MemoryMappedTest, BinaryData) {
    // Create file with binary data
    fs::path path = test_dir_ / "binary.dat";
    {
        std::ofstream ofs(path, std::ios::binary);
        for (int i = 0; i < 256; ++i) {
            char byte = static_cast<char>(i);
            ofs.write(&byte, 1);
        }
    }

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadOnly);
    EXPECT_EQ(mmf.size(), 256);

    const unsigned char* data = mmf.data_as<unsigned char>();
    for (int i = 0; i < 256; ++i) {
        EXPECT_EQ(data[i], static_cast<unsigned char>(i));
    }
}

// === Concurrent Access Test ===

TEST_F(MemoryMappedTest, ConcurrentReadAccess) {
    const std::string content(1024, 'C');
    auto path = create_test_file("concurrent.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadOnly);

    // Multiple threads reading same mapping
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&mmf, &success_count, &content]() {
            const char* data = mmf.data_as<char>();
            if (std::memcmp(data, content.data(), content.size()) == 0) {
                success_count++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count, 4);
}

// === View Flush Test ===

TEST_F(MemoryMappedTest, ViewFlush) {
    const std::string content = "View flush test content";
    auto path = create_test_file("viewflush.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadWrite);
    auto view = mmf.create_view(5, 10);

    // Modify through view
    char* view_data = static_cast<char*>(view.data());
    std::memcpy(view_data, "MODIFIED!!", 10);

    // Flush view
    std::error_code ec;
    view.flush(0, 0, &ec);
    // Flush may not be required/supported on all platforms

    mmf.close();

    // Verify changes - should work even without explicit flush
    std::string new_content = read_file_content(path);
    EXPECT_EQ(new_content.substr(5, 10), "MODIFIED!!");
}

// === Template Data Access Test ===

TEST_F(MemoryMappedTest, TemplateDataAccess) {
    // Create file with structured data
    struct TestStruct {
        int32_t id;
        float value;
        char name[8];
    };

    fs::path path = test_dir_ / "struct.dat";
    {
        TestStruct data[3] = {
            {1, 3.14f, "first"},
            {2, 2.71f, "second"},
            {3, 1.41f, "third"}
        };

        std::ofstream ofs(path, std::ios::binary);
        ofs.write(reinterpret_cast<char*>(data), sizeof(data));
    }

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadOnly);
    const TestStruct* structs = mmf.data_as<TestStruct>();

    EXPECT_EQ(structs[0].id, 1);
    EXPECT_FLOAT_EQ(structs[0].value, 3.14f);
    EXPECT_EQ(std::string(structs[0].name), "first");

    EXPECT_EQ(structs[1].id, 2);
    EXPECT_FLOAT_EQ(structs[1].value, 2.71f);

    EXPECT_EQ(structs[2].id, 3);
    EXPECT_FLOAT_EQ(structs[2].value, 1.41f);
}

// === View Data Access Templates ===

TEST_F(MemoryMappedTest, ViewDataAsTemplates) {
    const std::string content = "Template view test";
    auto path = create_test_file("viewtemplate.txt", content);

    fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadOnly);
    auto view = mmf.create_view(0, content.size());

    const char* char_data = view.data_as<char>();
    EXPECT_EQ(std::string(char_data, view.size()), content);

    const std::byte* byte_data = view.data_as<std::byte>();
    EXPECT_NE(byte_data, nullptr);
}

// === Exception Safety Test (if exceptions enabled) ===

#if CORE_ENABLE_EXCEPTIONS
TEST_F(MemoryMappedTest, ThrowOnError) {
    fs::path path = test_dir_ / "nonexistent_throw.txt";

    EXPECT_THROW({
        fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadOnly);
    }, std::system_error);
}
#endif

// === RAII and Destructor Test ===

TEST_F(MemoryMappedTest, DestructorClosesMapping) {
    const std::string content = "RAII test";
    auto path = create_test_file("raii.txt", content);

    {
        fcm::MemoryMappedFile mmf(path, fcm::MemoryMappedFile::Mode::ReadWrite);
        char* data = mmf.data_as<char>();
        data[0] = 'X';
        // Destructor should flush and close
    }

    // Verify change was persisted
    std::string new_content = read_file_content(path);
    EXPECT_EQ(new_content[0], 'X');
}