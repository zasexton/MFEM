#pragma once

#ifndef CORE_MEMORY_INTRUSIVE_LIST_H
#define CORE_MEMORY_INTRUSIVE_LIST_H

#include <cstddef>
#include <iterator>

namespace fem::core::memory {

struct intrusive_list_node {
    intrusive_list_node* prev{nullptr};
    intrusive_list_node* next{nullptr};
    bool linked() const noexcept { return prev != nullptr || next != nullptr; }
};

template<class T, intrusive_list_node T::* Hook>
class intrusive_list {
public:
    intrusive_list() = default;
    ~intrusive_list() { clear(); }

    intrusive_list(const intrusive_list&) = delete;
    intrusive_list& operator=(const intrusive_list&) = delete;

    class iterator {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = T*;
        using reference = T&;
        using iterator_category = std::bidirectional_iterator_tag;

        iterator() = default;
        explicit iterator(intrusive_list_node* n) : n_(n) {}

        reference operator*() const { return *node_to_obj(n_); }
        pointer operator->() const { return node_to_obj(n_); }
        iterator& operator++() { n_ = n_->next; return *this; }
        iterator operator++(int) { iterator t(*this); ++(*this); return t; }
        iterator& operator--() { n_ = n_->prev; return *this; }
        iterator operator--(int) { iterator t(*this); --(*this); return t; }
        bool operator==(const iterator& o) const { return n_ == o.n_; }
        bool operator!=(const iterator& o) const { return n_ != o.n_; }

    private:
        intrusive_list_node* n_{nullptr};
        static pointer node_to_obj(intrusive_list_node* n) {
            // Recover T* from node pointer using Hook offset
            char* base = reinterpret_cast<char*>(n);
            char* obj = base - offset_of();
            return reinterpret_cast<pointer>(obj);
        }
        static constexpr std::ptrdiff_t offset_of() {
            // Compute offset of Hook within T
            const T* t = reinterpret_cast<const T*>(0x1000);
            const auto member_ptr = &(t->*Hook);
            return reinterpret_cast<const char*>(member_ptr) - reinterpret_cast<const char*>(t);
        }
        friend class intrusive_list;
    };

    [[nodiscard]] bool empty() const noexcept { return head_.next == &tail_; }
    [[nodiscard]] std::size_t size() const noexcept {
        std::size_t c = 0;
        const intrusive_list_node* node = head_.next;
        while (node != &tail_) {
            ++c;
            node = node->next;
        }
        return c;
    }

    iterator begin() { return iterator(head_.next); }
    iterator end() { return iterator(&tail_); }

    iterator begin() const { return iterator(const_cast<intrusive_list_node*>(head_.next)); }
    iterator end() const { return iterator(const_cast<intrusive_list_node*>(&tail_)); }

    void push_front(T& obj) { insert_after(&head_, node_of(obj)); }
    void push_back(T& obj) { insert_before(&tail_, node_of(obj)); }

    void erase(T& obj) { unlink(node_of(obj)); }
    void clear() { while (!empty()) erase(*begin()); }

private:
    intrusive_list_node head_{nullptr, &tail_};
    intrusive_list_node tail_{&head_, nullptr};

    static intrusive_list_node* node_of(T& obj) { return &(obj.*Hook); }

    static void insert_after(intrusive_list_node* pos, intrusive_list_node* node) {
        node->prev = pos;
        node->next = pos->next;
        pos->next->prev = node;
        pos->next = node;
    }
    static void insert_before(intrusive_list_node* pos, intrusive_list_node* node) { insert_after(pos->prev, node); }
    static void unlink(intrusive_list_node* node) {
        if (!node->prev && !node->next) return; // not linked
        node->prev->next = node->next;
        node->next->prev = node->prev;
        node->prev = node->next = nullptr;
    }
};

} // namespace fem::core::memory

#endif // CORE_MEMORY_INTRUSIVE_LIST_H

