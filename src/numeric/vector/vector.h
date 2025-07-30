#pragma once

#ifndef VECTOR_H
#define VECTOR_H

#include <array>
#include <memory>
#include <cstddef>
#include <cstring>
#include <utility>
#include <type_traits>
#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <stdexcept>

#include "numeric/scalar/traits.h"
#include "numeric/scalar/concepts.h"
#include "numeric/vector/traits.h"

namespace numeric::vector {

    template <typename T, std::size_t N>
    requires (numeric::scalar::NumberLike<T>)
    class Vector<T, N, Dense>
    {
    public:
        using value_type      = T;
        using size_type       = std::size_t;
        using reference       = value_type&;
        using const_reference = const value_type&;
        using iterator        = value_type*;
        using const_iterator  = const value_type*;

        static constexpr size_type static_size = N;
        static_assert(static_size != Dynamic,
                      "Primary template instantiated with Dynamic; "
                      "use the dynamic specialization.");

        // constructors
        constexpr Vector() = default;
        constexpr explicit Vector(std::initializer_list<T> list)
        {
            assert(list.size() == N && "Initializer_list size mismatch");
            std::copy_n(list.begin(), N, m_data.begin());
        }

        constexpr explicit Vector(const T& v)
        {
            std::fill_n(m_data.begin(), N, v);
        }

        [[nodiscard]] constexpr reference operator[](size_type i) noexcept
        {
            assert(i < N);
            return m_data[i];
        }

        [[nodiscard]] constexpr const_reference operator[](size_type i) const noexcept
        {
            assert(i < N);
            return m_data[i];
        }

        [[nodiscard]] constexpr reference at(size_type i)
        {
            if (i >= N) throw std::out_of_range{"numeric::vector::Vector::at"};
            return m_data[i];
        }

        [[nodiscard]] constexpr const_reference at(size_type i) const
        {
            if (i >= N) throw std::out_of_range{"numeric::vector::Vector::at"};
            return m_data[i];
        }

        // Size / iterator

        [[nodiscard]] static constexpr size_type size() noexcept {return N;}
        [[nodiscard]] static constexpr bool empty() noexcept { return false;}
        [[nodiscard]] static constexpr size_type nnz() noexcept { return N; }

        [[nodiscard]] constexpr iterator begin() noexcept { return m_data.data(); }
        [[nodiscard]] constexpr const_iterator begin() const noexcept { return m_data.data(); }
        [[nodiscard]] constexpr iterator end() noexcept { return m_data.data() + N; }
        [[nodiscard]] constexpr const_iterator end() const noexcept { return m_data.data() + N;}

        // Raw data point
        [[nodiscard]] constexpr value_type* data() noexcept { return m_data.data(); }
        [[nodiscard]] constexpr const value_type* data() const noexcept { return m_data.data(); }

        // Swap & equality
        friend constexpr void swap(Vector& a, Vector& b) noexcept
        {
            using std::swap;
            swap(a.m_data, b.m_data);
        }

        friend constexpr bool operator==(const Vector& a, const Vector& b) noexcept
        {
            return a.m_data == b.m_data;
        }

        friend constexpr bool operator!=(const Vector& a, const Vector& b) noexcept
        {
            return a.m_data != b.m_data;
        }
    private:
        std::array<T, N> m_data{};
    };

    template <typename T, std::size_t N>
    requires (numeric::scalar::NumberLike<T>)
    class Vector<T, N, Sparse>
    {
    public:
        using value_type      = T;
        using index_type      = std::size_t;
        using size_type       = std::size_t;
        using reference       = value_type&;
        using const_reference = const value_type&;

        static constexpr size_type static_size = N;
        static constexpr size_type capacity    = N;
        static_assert(static_size != Dynamic, "Vector<T,Dynamic,Sparse> lives in vector.h");

        struct nz_iterator {
            index_type* ip;
            value_type* vp;

            auto operator*() const noexcept -> std::pair<index_type&, value_type&>
            {
                return { *ip, *vp };
            }

            nz_iterator& operator++() noexcept {
                ++ip; ++vp; return *this;
            }

            bool operator!=(const nz_iterator& other) const noexcept {
                return ip != other.ip;
            }
        };


        // Constructors

        constexpr Vector() = default;

        constexpr Vector(std::initializer_list<std::pair<index_type,value_type>> nz)
        {
            assert(nz.size() <= capacity && "too many non-zeros for static sparse vector.");
            for (auto [i,v] : nz) {
                push_back(i,v);
            }
            sort_and_unique();
        }

        // --------------------  element access  --------------------------
        [[nodiscard]] constexpr reference operator[](size_type i) noexcept
        {
            if (i >= N) {
                throw std::out_of_range{"numeric::vector::Vector<Sparse>::at"};
            }
            auto pos = find_or_emplace(i);
            return m_val[pos];
        }

        [[nodiscard]] constexpr value_type operator[](index_type i) const noexcept
        {
            assert(i < N);
            auto pos = find_index(i);
            return (pos == npos) ? value_type{} : m_val[pos];
        }

        constexpr reference at(index_type i)
        {
            if (i >= N) {
                throw std::out_of_range{"numeric::vector::Vector<Sparse>::at"};
            }
            auto pos = find_or_emplace(i);
            return m_val[pos];
        }

        [[nodiscard]] constexpr const_reference at(index_type i) const
        {
            if (i>=N) {
                throw std::out_of_range{"numeric::vector::Vector<Sparse>::at"};
            }
            auto pos = find_index(i);
            if (pos == npos) {
                throw std::out_of_range{"numeric::vector::Vector<Sparse>::at (zero)"};
            }
            return m_val[pos];
        }

        // ---------------------  modifiers  ------------------------------
        constexpr void push_back(index_type i, const value_type& v)
        {
            assert(i < N && "index out of bounds");
            assert(m_nnz < capacity && "static sparse vector capacity exhausted");
            m_idx[m_nnz] = i;
            m_val[m_nnz] = v;
            ++m_nnz;
        }

        /// insert-or-assign; keeps the array sorted
        constexpr void set(index_type i, const value_type& v)
        {
            auto pos = find_or_emplace(i);
            m_val[pos] = v;
        }

        // --------------------  size / iteration  ------------------------
        [[nodiscard]] static constexpr size_type size()  noexcept { return static_size; }
        [[nodiscard]] constexpr size_type        nnz()   const noexcept { return m_nnz;  }
        [[nodiscard]] static constexpr bool      empty() noexcept { return false; }

        struct nz_const_iterator
        {
            const index_type* ip;
            const value_type* vp;
            auto operator*() const { return std::pair{*ip,*vp}; }
            nz_const_iterator& operator++() { ++ip; ++vp; return *this; }
            bool operator!=(nz_const_iterator other) const { return ip != other.ip; }
        };
        [[nodiscard]] constexpr nz_const_iterator begin() const
        { return { m_idx.data(), m_val.data() }; }
        [[nodiscard]] constexpr nz_const_iterator end() const
        { return { m_idx.data()+m_nnz, m_val.data()+m_nnz }; }
        [[nodiscard]] nz_iterator begin() noexcept {
            return { m_idx.data(), m_val.data() };
        }
        [[nodiscard]] nz_iterator end() noexcept {
            return { m_idx.data() + m_nnz, m_val.data() + m_nnz };
        }

        // ----------------------  raw data  ------------------------------
        [[nodiscard]] constexpr const index_type* indices() const noexcept { return m_idx.data(); }
        [[nodiscard]] constexpr const value_type* values () const noexcept { return m_val.data(); }

        // ------------------  comparison / swap  -------------------------
        friend constexpr bool operator==(const Vector& a, const Vector& b) noexcept
        {
            return a.m_nnz == b.m_nnz &&
                   std::equal(a.m_idx.begin(), a.m_idx.begin()+a.m_nnz, b.m_idx.begin()) &&
                   std::equal(a.m_val.begin(), a.m_val.begin()+a.m_nnz, b.m_val.begin());
        }

        friend constexpr bool operator!=(const Vector& a, const Vector& b) noexcept
        { return !(a==b); }

        friend constexpr void swap(Vector& a, Vector& b) noexcept
        {
            using std::swap;
            const auto nnz_max = std::max(a.m_nnz, b.m_nnz);
            for (size_type k=0;k<nnz_max;++k) {
                swap(a.m_idx[k], b.m_idx[k]);
                swap(a.m_val[k], b.m_val[k]);
            }
            swap(a.m_nnz, b.m_nnz);
        }

    private:
        static constexpr size_type npos = static_cast<size_type>(-1);
        // -------------------  private methods  -----------------------------
        constexpr size_type find_index(index_type i) const noexcept
        {
            auto it = std::lower_bound(m_idx.begin(), m_idx.begin()+m_nnz,i);
            return (it != m_idx.begin()+m_nnz && *it == i) ? std::distance(m_idx.begin(),it) : npos;
        }

        constexpr size_type find_or_emplace(index_type i)
        {
            auto it = std::lower_bound(m_idx.begin(), m_idx.begin()+m_nnz, i);
            size_type pos = std::distance(m_idx.begin(), it);
            if (it == m_idx.begin()+m_nnz || *it != i) {
                assert(m_nnz < capacity && "static sparse vector capacity exhausted");
                // shift right by 1
                for (size_type k=m_nnz; k>pos; --k) {
                    m_idx[k] = m_idx[k-1];
                    m_val[k] = m_val[k-1];
                }
                m_idx[pos] = i;
                m_val[pos] = value_type{};    // default initialise
                ++m_nnz;
            }
            return pos;
        }

        constexpr void sort_and_unique()
        {
            // Simple insertion sort – list is tiny for static N
            for (size_type i=1;i<m_nnz;++i)
            {
                index_type key_i = m_idx[i];
                value_type key_v = m_val[i];
                size_type j=i;
                while (j>0 && m_idx[j-1] > key_i) {
                    m_idx[j] = m_idx[j-1];
                    m_val[j] = m_val[j-1];
                    --j;
                }
                if (j>0 && m_idx[j-1]==key_i) {      // duplicate index – last one wins
                    m_val[j-1] = key_v;
                } else {
                    m_idx[j] = key_i;
                    m_val[j] = key_v;
                }
            }
        }
        // -------------------  data members  -----------------------------
        std::array<index_type, capacity> m_idx{};
        std::array<value_type, capacity> m_val{};
        size_type                        m_nnz{0};
    };

    // Partial specialization for dynamic vectors
    template <typename T>
    requires (numeric::scalar::NumberLike<T>)
    class Vector<T, Dynamic, Dense>
    {
    public:
        using value_type        = T;
        using size_type         = std::size_t;
        using reference         = value_type&;
        using const_reference   = const value_type&;
        using iterator          = value_type*;
        using const_iterator    = const value_type*;

        static constexpr size_type static_size = Dynamic;

        // Constructors
        constexpr Vector() = default;

        // Construct dynamic vector with explicit length, default-initialized
        explicit Vector(size_type n) : m_size{n}, m_data{n ? std::make_unique<value_type[]>(n) : nullptr}
        {}

        // Construct & fill with value
        Vector(size_type n, const value_type& v)
            : Vector(n)
        {
            std::fill_n(m_data.get(), m_size, v);
        }

        // Construct from initializer list
        explicit Vector(std::initializer_list<T> list)
            : Vector(list.size())
        {
            std::copy(list.begin(), list.end(), m_data.get());
        }

        // Copy constructor
        Vector(const Vector& other)
            : Vector(other.m_size)
        {
            std::copy_n(other.m_data.get(), m_size, m_data.get());
        }

        // Move Constructor
        Vector(Vector&& other) noexcept
            : m_size(other.m_size), m_data(std::move(other.m_data))
        {
            other.m_size = 0;
        }

        // Copy assignment
        Vector& operator=(const Vector& other)
        {
            using std::swap;
            if (this == &other) return *this;
            Vector tmp(other);
            swap(*this, tmp);
            return *this;
        }

        // Move assignment
        Vector& operator=(Vector&& other) noexcept
        {
            if (this != &other) {
                m_size = other.m_size;
                m_data = std::move(other.m_data);
                other.m_size = 0;
            }
            return *this;
        }

        // Element access
        [[nodiscard]] reference operator[](size_type i) noexcept
        {
            assert(i < m_size);
            return m_data[i];
        }

        [[nodiscard]] const_reference operator[](size_type i) const noexcept
        {
            assert(i < m_size);
            return m_data[i];
        }

        [[nodiscard]] reference at(size_type i)
        {
            if (i >= m_size) throw std::out_of_range{"numeric::vector::Vector::at"};
            return m_data[i];
        }

        [[nodiscard]] const_reference at(size_type i) const
        {
            if (i >= m_size) throw std::out_of_range{"numeric::vector::Vector::at"};
            return m_data[i];
        }

        // Size / iterators
        [[nodiscard]] constexpr size_type size() const noexcept { return m_size; }
        [[nodiscard]] constexpr bool      empty() const noexcept { return m_size == 0;}
        [[nodiscard]] constexpr size_type nnz() const noexcept { return m_size; }

        [[nodiscard]] iterator            begin() noexcept { return m_data.get(); }
        [[nodiscard]] const_iterator      begin() const noexcept { return m_data.get(); }
        [[nodiscard]] iterator            end()   noexcept { return m_data.get() + m_size;}
        [[nodiscard]] const_iterator      end() const noexcept { return m_data.get() + m_size;}

        [[nodiscard]] value_type*         data() noexcept       { return m_data.get(); }
        [[nodiscard]] const value_type*   data() const noexcept { return m_data.get(); }

        // Resize

        void resize(size_type new_size)
        {
            using std::swap;
            if (new_size == m_size) return;
            Vector tmp(new_size);
            const size_type copy_count = (new_size < m_size) ? new_size : m_size;
            std::copy_n(m_data.get(), copy_count, tmp.m_data.get());
            swap(*this, tmp);
        }


        // Swap & equality
        friend void swap(Vector& a, Vector& b) noexcept
        {
            using std::swap;
            swap(a.m_size, b.m_size);
            swap(a.m_data, b.m_data);
        }

        friend bool operator==(const Vector& a, const Vector& b) noexcept
        {

            if (a.m_size != b.m_size) return false;

            if constexpr (std::is_trivially_copyable_v<T>)
            {
                return std::memcmp(a.m_data.get(), b.m_data.get(),
                                   a.m_size * sizeof(T)) == 0;
            } else {
                return std::equal(a.begin(), a.end(), b.begin());
            }
        }

        friend bool operator!=(const Vector& a, const Vector& b) noexcept
        {
            return !(a == b);
        }
    private:
        size_type                     m_size{0};
        std::unique_ptr<value_type[]> m_data {nullptr};
    };

    template <typename T>
    requires (numeric::scalar::NumberLike<T>)
    class Vector<T, Dynamic, Sparse>
    {
        // ---------- public member types ----------
    public:
        using value_type      = T;
        using index_type      = std::size_t;
        using size_type       = std::size_t;
        using reference       = value_type&;
        using const_reference = const value_type&;

        static constexpr size_type static_size = Dynamic;

        struct nz_iterator {
            index_type* ip;
            value_type* vp;

            auto operator*() const noexcept -> std::pair<index_type&, value_type&>
            {
                return { *ip, *vp };
            }

            nz_iterator& operator++() noexcept {
                ++ip; ++vp; return *this;
            }

            bool operator!=(const nz_iterator& other) const noexcept {
                return ip != other.ip;
            }
        };


        // ---------- ctors ----------
        constexpr Vector() = default;

        explicit Vector(size_type n) : m_size(n) {}

        Vector(size_type n,
               std::initializer_list<std::pair<index_type,value_type>> nz)
                : m_size(n)
        {
            reserve(nz.size());
            for (auto [i,v] : nz) {
                push_back(i,v);
            }
            sort_and_unique();
        }

        // ---------- element access ----------
        [[nodiscard]] reference operator[](size_type i) noexcept
        {
            assert(i < m_size);
            auto pos = find_or_emplace(i);
            return m_val[pos];
        }

        [[nodiscard]] value_type operator[](index_type i) const noexcept
        {
            assert(i < m_size);
            auto pos = find_index(i);
            return (pos == npos) ? value_type{} : m_val[pos];
        }

        reference at(index_type i)                           // inserts if absent
        {
            if (i >= m_size) throw std::out_of_range{"Vector<Sparse>::at"};
            auto pos = find_or_emplace(i);
            return m_val[pos];
        }
        const_reference at(index_type i) const
        {
            if (i >= m_size) throw std::out_of_range{"Vector<Sparse>::at"};
            auto pos = find_index(i);
            if (pos == npos) throw std::out_of_range{"Vector<Sparse>::at (zero)"};
            return m_val[pos];
        }

        // ---------- size / iterators ----------
        [[nodiscard]] constexpr size_type size()  const noexcept { return m_size; }
        [[nodiscard]] constexpr bool     empty() const noexcept { return m_size == 0; }
        [[nodiscard]] size_type          nnz()   const noexcept { return m_val.size(); }

        struct nz_const_iterator {
            const index_type* ip;
            const value_type* vp;
            auto operator*() const { return std::pair{*ip,*vp}; }
            nz_const_iterator& operator++() { ++ip; ++vp; return *this; }
            bool operator!=(nz_const_iterator rhs) const { return ip != rhs.ip; }
        };

        nz_const_iterator begin() const { return {m_idx.data(), m_val.data()}; }
        nz_const_iterator end()   const { return {m_idx.data()+nnz(), m_val.data()+nnz()}; }
        [[nodiscard]] nz_iterator begin() noexcept {
            return { m_idx.data(), m_val.data() };
        }
        [[nodiscard]] nz_iterator end() noexcept {
            return { m_idx.data() + m_size, m_val.data() + m_size };
        }

        // ---------- modifiers ----------
        void reserve(size_type nnz) {
            m_idx.reserve(nnz);
            m_val.reserve(nnz);
        }

        void push_back(index_type i, const value_type& v)
        {
            assert(i < m_size && "index out of bounds");
            m_idx.push_back(i);
            m_val.push_back(v);
        }

        void set(index_type i, const value_type& v) { at(i) = v; }

        void resize(size_type new_size)
        {
            m_size = new_size;
            // drop out-of-range entries
            auto first_bad = std::lower_bound(m_idx.begin(), m_idx.end(), new_size);
            auto drop = std::distance(first_bad, m_idx.end());
            m_idx.erase(first_bad, m_idx.end());
            m_val.erase(m_val.end() - drop, m_val.end());
        }

        // ---------- swap / equality ----------
        friend void swap(Vector& a, Vector& b) noexcept
        {
            using std::swap;
            swap(a.m_size, b.m_size);
            swap(a.m_idx , b.m_idx);
            swap(a.m_val , b.m_val);
        }

        friend bool operator==(const Vector& a, const Vector& b) noexcept
        {
            return a.m_size == b.m_size &&
                   a.m_idx  == b.m_idx  &&
                   a.m_val  == b.m_val;
        }

        friend bool operator!=(const Vector& a, const Vector& b) noexcept
        { return !(a==b); }

    private:
        // ---------- helpers ----------
        static constexpr size_type npos = static_cast<size_type>(-1);

        size_type find_index(index_type i) const noexcept
        {
            auto it = std::lower_bound(m_idx.begin(), m_idx.end(), i);
            return (it != m_idx.end() && *it == i) ?
                   std::distance(m_idx.begin(), it) : npos;
        }

        size_type find_or_emplace(index_type i)
        {
            auto it  = std::lower_bound(m_idx.begin(), m_idx.end(), i);
            size_type pos = std::distance(m_idx.begin(), it);
            if (it == m_idx.end() || *it != i) {
                m_idx.insert(it, i);
                m_val.insert(m_val.begin()+pos, value_type{});
            }
            return pos;
        }

        void sort_and_unique()
        {
            std::vector<size_type> p(m_idx.size());
            std::iota(p.begin(), p.end(), 0);
            std::sort(p.begin(), p.end(),
                      [&](auto a, auto b){ return m_idx[a] < m_idx[b]; });
            reorder(p, m_idx);
            reorder(p, m_val);
            // collapse duplicates (last value wins)
            auto out = 0UL;
            for (size_type in=0; in<m_idx.size(); ++in) {
                if (out && m_idx[in]==m_idx[out-1]) { m_val[out-1] = m_val[in]; }
                else {
                    if (out!=in) { m_idx[out]=m_idx[in]; m_val[out]=m_val[in]; }
                    ++out;
                }
            }
            m_idx.resize(out);
            m_val.resize(out);
        }

        template<numeric::scalar::NumberLike U>
        static void reorder(const std::vector<size_type>& p, std::vector<U>& v)
        {
            std::vector<U> tmp(v.size());
            for (size_type k=0;k<p.size();++k) {
                tmp[k] = std::move(v[p[k]]);
            }
            v.swap(tmp);
        }

        // ---------- data members ----------
        size_type               m_size{0};
        std::vector<index_type> m_idx;
        std::vector<value_type> m_val;
    };
}

#endif //VECTOR_H
