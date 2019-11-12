#pragma once
// Contains vocabulary types, basic functions used in many places.

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

/* Use external printf C++ library : fmtlib.
 * From https://github.com/fmtlib/fmt
 * Documentation is available at http://fmtlib.net/latest/index.html
 * Format string syntax at http://fmtlib.net/latest/syntax.html
 *
 * The standard C++ printf equivalent is <iostream>.
 * But iostreams are very different from printf, and not compatible with FILE*.
 * Specific number formatting is difficult to do (internal state, <iomanip>).
 * Performance is not very good.
 *
 * I decided to use fmtlib which is a nice replacement for iostream.
 * This is an external library, header-only (only .h files).
 * Files have been included in the repository in the external/ directory.
 * Files are from the release version 5.3.0.
 *
 * Usage of the library:
 * void fmt::print (FILE* output, "format string", ...); // for stdout, stderr, ...
 * std::string fmt::format("format_string", ...); // to generate a std::string.
 */
//#define FMT_HEADER_ONLY 1 // In header only mode
#include "external/fmt/format.h"
#include "external/fmt/ostream.h"

/*******************************************************************************
 * Optionally stores a value.
 * Simpler but less capable version of C++17 std::optional<T>.
 * T must be default constructible.
 */
template <typename T> struct Optional {
    T value{};
    bool has_value{false};

    constexpr Optional() = default; // No value
    constexpr Optional(T t) : value(t), has_value(true) {}

    constexpr Optional & operator=(T t) {
        value = t;
        has_value = true;
        return *this;
    }

    // Test if we have a value
    constexpr operator bool() const noexcept { return has_value; }
};

/*******************************************************************************
 * span<T> represents a reference to a segment of a T array.
 * Equivalent to a (T* base, int len).
 */
template <typename T> struct span {
    T * base_{nullptr};
    std::size_t size_{0};

    // Build span from (ptr, size) or (ptr, ptr).
    span(T * base, std::size_t size) : base_(base), size_(size) {}
    span(T * base, T * end) : span(base, end - base) {}
    // Allow conversion of span<U> to span<T> if U* converts to T*
    template <typename U> span(span<U> s) : span(s.begin(), s.size()) {}

    // Accessors
    T * begin() const { return base_; }
    T * end() const { return base_ + size_; }
    std::size_t size() const { return size_; }
    T & operator[](std::size_t index) const {
        assert(index < size());
        return base_[index];
    }
};

// Create span from data structures
template <typename T> span<const T> make_span(const std::vector<T> & vec) {
    return {vec.data(), vec.size()};
}
template <typename T> span<T> make_span(std::vector<T> & vec) {
    return {vec.data(), vec.size()};
}
template <typename T, std::size_t N> span<const T> make_span(const T (&array)[N]) {
    return {&array[0], N};
}
template <typename T, std::size_t N> span<T> make_span(T (&array)[N]) {
    return {&array[0], N};
}
template <typename T, std::size_t N> span<const T> make_span(const std::array<T, N> & a) {
    return {&a[0], N};
}
template <typename T, std::size_t N> span<T> make_span(std::array<T, N> & a) {
    return {&a[0], N};
}
template <typename T> span<T> make_span(T * base, T * end) {
    return {base, end};
}
template <typename T> span<T> make_span(T * base, std::size_t size) {
    return {base, size};
}

// Slice (sub span)
template <typename T> span<T> slice(span<T> s, std::size_t from, std::size_t size) {
    return {s.begin() + from, size};
}
template <typename T> span<T> slice_from(span<T> s, std::size_t from) {
    return {s.begin() + from, s.end()};
}

/*******************************************************************************
 * string_view: Const reference (pointer) to a sequence of char.
 * Does not own the data, only points to it.
 * The char sequence may not be null terminated.
 *
 * std::string_view exists in C++17, but not before sadly.
 * fmtlib provides a basic incomplete implementation that we reuse here.
 * This implementation is replaced by an alias to std::string_view in C++17 (with #ifdef).
 * Some functions are provided to add back missing functionnality.
 */
using string_view = fmt::string_view;

inline string_view make_string_view(const char * begin, const char * end) {
    return string_view(begin, end - begin);
}
inline bool empty(string_view sv) {
    return sv.size() == 0;
}
inline std::string to_string(string_view sv) {
    return std::string(sv.begin(), sv.end());
}

// Prefix / Suffix
inline bool starts_with(string_view prefix, string_view str) {
    if(prefix.size() <= str.size()) {
        return std::equal(prefix.begin(), prefix.end(), str.begin());
    } else {
        return false;
    }
}
inline bool starts_with(char prefix, string_view str) {
    return str.size() >= 1 && str.data()[0] == prefix;
}

inline bool ends_with(string_view suffix, string_view str) {
    if(suffix.size() <= str.size()) {
        return std::equal(suffix.begin(), suffix.end(), str.end() - suffix.size());
    } else {
        return false;
    }
}

// Remove whitespace (using C locale only)
inline bool is_space(char c) {
    return std::isspace(static_cast<int>(c));
}
inline string_view trim_ws_left(string_view str) {
    auto after_skipped_space = std::find_if_not(str.begin(), str.end(), is_space);
    return make_string_view(after_skipped_space, str.end());
}
inline string_view trim_ws_right(string_view str) {
    using reverse_it = std::reverse_iterator<string_view::iterator>;
    auto last_skipped_space = std::find_if_not(reverse_it(str.end()), reverse_it(str.begin()), is_space).base();
    return make_string_view(str.begin(), last_skipped_space);
}
inline string_view trim_ws(string_view str) {
    return trim_ws_right(trim_ws_left(str));
}

// Split at separator. Does not remove empty parts, does not trim whitespace.
inline std::vector<string_view> split(char separator, string_view text) {
    std::vector<string_view> r;
    const auto end = text.end();
    string_view::iterator part_begin = text.begin();

    while(true) {
        auto part_end = std::find(part_begin, end, separator);
        r.emplace_back(make_string_view(part_begin, part_end));
        if(part_end == end) {
            break;
        }
        part_begin = part_end + 1; // Skip separator
    }
    return r;
}
// Like split above, but only get the first N parts, or empty optional if not enough parts.
template <std::size_t N> Optional<std::array<string_view, N>> split_first_n(char separator, string_view text) {
    std::array<string_view, N> parts;
    const auto end = text.end();
    string_view::iterator part_begin = text.begin();
    std::size_t i = 0;
    while(true) {
        auto part_end = std::find(part_begin, end, separator);
        parts[i] = make_string_view(part_begin, part_end);
        ++i;
        if(i == N) {
            return parts;
        }
        if(part_end == end) {
            return {}; // Not enough parts
        }
        part_begin = part_end + 1; // Skip separator
    }
}

// Parse numbers
inline double parse_double(string_view str, string_view what) {
    const auto null_terminated = to_string(str);
    char * end = nullptr;
    const double d = std::strtod(null_terminated.data(), &end);
    if(end == null_terminated.data()) {
        throw std::runtime_error(fmt::format("Unable to parse value for {} from '{}'", what, str));
    }
    return d;
}
inline double parse_strict_positive_double(string_view str, string_view what) {
    double d = parse_double(str, what);
    if(!(0 < d)) {
        throw std::runtime_error(fmt::format("Value for {} is not > 0: {:#.15g}", what, d));
    }
    return d;
}
inline long parse_int(string_view str, string_view what) {
    const auto null_terminated = to_string(str);
    char * end = nullptr;
    const long n = std::strtol(null_terminated.data(), &end, 0);
    if(end == null_terminated.data()) {
        throw std::runtime_error(fmt::format("Unable to parse integer value for {} from '{}'", what, str));
    }
    return n;
}
inline std::size_t parse_strict_positive_int(string_view str, string_view what) {
    auto n = parse_int(str, what);
    if(!(n > 0)) {
        throw std::runtime_error(fmt::format("Value for {} is not > 0: {}", what, n));
    }
    return std::size_t(n);
}

/*******************************************************************************
 * Dynamically sized 2d vector.
 * Linearized in memory: row by row.
 * array(row, col) = inner[row * nb_col + col]
 */
template <typename T> class Vector2d {
  private:
    std::vector<T> inner_;
    std::size_t nb_rows_{};
    std::size_t nb_cols_{};

    std::size_t linear_index(std::size_t row_index, std::size_t col_index) const {
        assert(row_index < nb_rows_);
        assert(col_index < nb_cols_);
        return row_index * nb_cols_ + col_index;
    }

  public:
    // Empty vector
    Vector2d() = default;

    // Sized vector
    Vector2d(std::size_t nb_rows, std::size_t nb_cols)
        : inner_(nb_rows * nb_cols), nb_rows_(nb_rows), nb_cols_(nb_cols) {}

    // Get sizes
    std::size_t nb_rows() const noexcept { return nb_rows_; }
    std::size_t nb_cols() const noexcept { return nb_cols_; }

    // Access cells directly
    const T & operator()(std::size_t row_index, std::size_t col_index) const noexcept {
        return inner_[linear_index(row_index, col_index)];
    }
    T & operator()(std::size_t row_index, std::size_t col_index) noexcept {
        return inner_[linear_index(row_index, col_index)];
    }

    // Access rows
    span<const T> row(std::size_t row_index) const noexcept {
        return slice(make_span(inner_), linear_index(row_index, 0), nb_cols_);
    }
    span<T> row(std::size_t row_index) noexcept {
        return slice(make_span(inner_), linear_index(row_index, 0), nb_cols_);
    }

    // Indexation operator[]: return row.
    span<const T> operator[](std::size_t row_index) const noexcept { return row(row_index); }
    span<T> operator[](std::size_t row_index) noexcept { return row(row_index); }

    // Add a row by copying elements from a span.
    void append_row(span<const T> values) {
        assert(values.size() == nb_cols_);
        inner_.insert(inner_.end(), values.begin(), values.end());
        ++nb_rows_;
    }
    // Add a row by moving elements from a span<T> (the span will point to moved-from T).
    void append_row_move(span<T> values) {
        assert(values.size() == nb_cols_);
        inner_.reserve(inner_.size() + nb_cols_);
        std::move(values.begin(), values.end(), std::back_inserter(inner_));
        ++nb_rows_;
    }
    // Add a row by moving elements (selected if given a r-value vector).
    void append_row(std::vector<T> && values) { append_row_move(make_span(values)); }
};

/*******************************************************************************
 * Vector<T> that keep its elements sorted and unique.
 */
template <typename T> class SortedVec {
  private:
    std::vector<T> inner;
    SortedVec(std::vector<T> && sorted_data) : inner(std::move(sorted_data)) {}

  public:
    SortedVec() = default;
    static SortedVec from_sorted(std::vector<T> && sorted_data) {
        const auto first_unsorted_or_duplicate_element = std::adjacent_find(
            sorted_data.begin(), sorted_data.end(), [](const T & current, const T & next) { return current >= next; });
        if(first_unsorted_or_duplicate_element != sorted_data.end()) {
            throw std::runtime_error("SortedVec::from_sorted: unsorted data");
        }
        return SortedVec(std::move(sorted_data));
    }
    static SortedVec from_unsorted(std::vector<T> && data) {
        std::sort(data.begin(), data.end());
        auto new_end = std::unique(data.begin(), data.end());
        data.erase(new_end, data.end());
        return SortedVec(std::move(data));
    }

    std::size_t size() const { return inner.size(); }
    const T & operator[](std::size_t i) const {
        assert(i < size());
        return inner[i];
    }

    using const_iterator = typename std::vector<T>::const_iterator;
    const_iterator begin() const { return inner.begin(); }
    const_iterator end() const { return inner.end(); }
};

/*******************************************************************************
 * Map to vector: apply a function to a range and put result in a vector.
 */
template <typename Range, typename Function> auto map_to_vector(const Range & range, Function f) {
    using ReturnType = decltype(f(*range.begin()));
    std::vector<ReturnType> vec;
    vec.reserve(range.size()); // Assume a vector like range.
    for(const auto & element : range) {
        vec.emplace_back(f(element));
    }
    return vec;
}

/*******************************************************************************
 * Power of 2 utils. For Haar base.
 */

// 2^n
inline size_t power_of_2(size_t n) {
    assert(n < size_t(std::numeric_limits<size_t>::digits));
    return size_t(1) << n;
}

// floor(log2(n)) for n > 0
inline size_t floor_log2(size_t n) {
    // iterative version : find higher 1 bit position in n.
    // maps to x86 builtin instruction in clang and -O3, no need for compiler builtins.
    assert(n > 0);
    size_t log = 0;
    while(n > 0) {
        n = n >> 1;
        log += 1;
    }
    return log - 1;
}