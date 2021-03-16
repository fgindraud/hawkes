#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <vector>

#include "utils.h"

/******************************************************************************
 * Index types.
 *
 * In general, all indexes use std::size_t to match usage of the C++ STL.
 * Provide light typedef for description of function APIs.
 * Strong typedefs are more verbose, and in general do not provide much safety in this case.
 */
using std::size_t;

using ProcessId = size_t;      // [0; nb_processes[
using FunctionBaseId = size_t; // [0; base_size[
using RegionId = size_t;       // [0; nb_regions[

/******************************************************************************
 * Process data.
 */

/* The space where points and point-related dimensional values live is the set of real numbers.
 * Points values are usually integers, but we need to be able to represent smaller than 1 distances.
 * Thus floating point numbers are used.
 * Point: a single point.
 * PointSpace: a distance, offset, intermediate value computed from Point positions.
 */
using Point = double;
using PointSpace = double;

// A point from input data : center + width of uncertainty
struct DataPoint {
    Point center;
    PointSpace width; // >= 0
};

inline bool operator<(const DataPoint & lhs, const DataPoint & rhs) {
    return lhs.center < rhs.center;
}
inline bool operator==(const DataPoint & lhs, const DataPoint & rhs) {
    return lhs.center == rhs.center && lhs.width == rhs.width;
}

template <typename T> struct DataByProcessRegion {
    Vector2d<T> data_; // Rows = regions, cols = processes.

    DataByProcessRegion(size_t nb_processes, size_t nb_regions) : data_(nb_regions, nb_processes) {
        assert(nb_processes > 0);
        assert(nb_regions > 0);
    }

    size_t nb_regions() const { return data_.nb_rows(); }
    size_t nb_processes() const { return data_.nb_cols(); }

    const T & data(ProcessId m, RegionId r) const { return data_(r, m); }
    T & data(ProcessId m, RegionId r) { return data_(r, m); }
    span<const T> data_for_region(RegionId r) const { return data_.row(r); }
};

/* Interval with configurable bounds.
 * This is a lightweight struct, it should be passed by value.
 */
enum class Bound { Open, Closed };

template <Bound bound_type> struct BoundCompare;
template <> struct BoundCompare<Bound::Open> {
    bool operator()(Point l, Point r) const noexcept { return l < r; }
};
template <> struct BoundCompare<Bound::Closed> {
    bool operator()(Point l, Point r) const noexcept { return l <= r; }
};

template <Bound lb, Bound rb> struct Interval {
    Point left{};
    Point right{};

    static constexpr Bound left_bound_type = lb;
    static constexpr Bound right_bound_type = rb;

    Interval() = default;
    Interval(PointSpace left_, PointSpace right_) : left(left_), right(right_) { assert(left <= right); }

    PointSpace width() const noexcept { return right - left; }
    Point center() const noexcept { return (left + right) / 2.; }
    bool in_left_bound(Point x) const noexcept { return BoundCompare<lb>()(left, x); }
    bool in_right_bound(Point x) const noexcept { return BoundCompare<rb>()(x, right); }
    bool contains(Point x) const noexcept { return BoundCompare<lb>()(left, x) && BoundCompare<rb>()(x, right); }
};

template <Bound lb, Bound rb> inline Interval<lb, rb> operator+(PointSpace offset, Interval<lb, rb> i) {
    return {i.left + offset, i.right + offset};
}
template <Bound lb, Bound rb> inline Interval<rb, lb> operator-(Interval<lb, rb> i) {
    return {-i.right, -i.left};
}
template <Bound lb, Bound rb> inline bool operator==(Interval<lb, rb> lhs, Interval<lb, rb> rhs) {
    return lhs.left == rhs.left && lhs.right == rhs.right;
}
template <Bound lb, Bound rb> inline Interval<lb, rb> union_(Interval<lb, rb> lhs, Interval<lb, rb> rhs) {
    return {std::min(lhs.left, rhs.left), std::max(lhs.right, rhs.right)};
}

/******************************************************************************
 * Function bases.
 */
struct Base {
    virtual ~Base() = default;

    // User printable name
    virtual std::string name() const = 0;
    // Base description generated in output file if verbose mode is used
    virtual void write_verbose_description(FILE * out) const = 0;
};

/* Histogram(base_size, D) : shifted indicator functions, with L2-norm of 1
 * For k in [0, base_size[ : phi_k(x) = 1/sqrt(D) * Indicator_]k * D, (k + 1) * D](x)
 */
struct HistogramBase final : Base {
    size_t base_size; // [1, inf[
    PointSpace delta; // ]0, inf[

    double normalization_factor;

    HistogramBase(size_t base_size_, PointSpace delta_) : base_size(base_size_), delta(delta_) {
        assert(base_size > 0);
        assert(delta > 0.);
        normalization_factor = 1. / std::sqrt(delta);
    }

    Interval<Bound::Open, Bound::Closed> total_span() const { return {0., PointSpace(base_size) * delta}; }

    struct Histogram {
        Interval<Bound::Open, Bound::Closed> interval;
        double normalization_factor;
    };
    Histogram histogram(FunctionBaseId k) const {
        assert(k < base_size);
        return {
            {PointSpace(k) * delta, PointSpace(k + 1) * delta},
            normalization_factor,
        };
    }

    std::string name() const final { return "histogram base"; }
    void write_verbose_description(FILE * out) const final {
        fmt::print(
            out,
            "# base = Histogram(K = {}, delta = {}) = {{\n"
            "#   phi_k = 1/sqrt(delta) * 1_]k*delta, (k+1)*delta] for k in [0,K[\n"
            "# }}\n",
            base_size,
            delta);
    }
};

/** Haar(nb_scales, delta) : haar square wavelets, L2-norm of 1
 * For:
 * s = scale in [0, nb_scales[
 * p = position in [0, 2^s[
 * delta > 0, so that base support is ]0, delta].
 * This is the set of function f_{s,p}(x) = sqrt(2)^s/sqrt(delta) * (
 *   Indicator_] delta * 2p / 2^(s+1), delta * (2p + 1) / 2^(s+1) ](x) -
 *   Indicator_] delta * (2p + 1) / 2^(s+1), delta * (2p + 2) / 2^(s+1) ](x)
 * )
 *
 * Mapping to phi_k:
 * Scale s has 2^s functions, so for [0, nb_scales[ : base_size = sum_s 2^s = 2^nb_scales - 1.
 * For k in [0, 2^nb_scales - 1[ : phi_k = g_{s,p} with 2^s + p == k + 1
 * Thus for a given k, s = floor(log2(k + 1)) and p = k + 1 - 2^s
 */
struct HaarBase final : Base {
    size_t nb_scales; // [1, inf[
    PointSpace delta; // ]0, inf[

    double delta_normalization_factor;

    // Cannot represent more than max_nb_scales due to base_size = 2^nb_scales - 1.
    // High scaling number should not be used due to computation requirements anyway.
    static constexpr size_t max_nb_scales = size_t(std::numeric_limits<FunctionBaseId>::digits - 1);

    HaarBase(size_t nb_scales_, PointSpace delta_) : nb_scales(nb_scales_), delta(delta_) {
        assert(nb_scales > 0);
        assert(nb_scales < max_nb_scales);
        assert(delta > 0.);
        delta_normalization_factor = 1. / std::sqrt(delta);
    }

    size_t base_size() const { return power_of_2(nb_scales) - 1; }

    Interval<Bound::Open, Bound::Closed> total_span() const { return {0., delta}; }

    struct ScalePosition {
        size_t scale;
        size_t position;
    };
    ScalePosition scale_and_position(FunctionBaseId k) const {
        assert(k < base_size());
        const size_t scale = floor_log2(k + 1);
        const size_t position = k + 1 - power_of_2(scale);
        return {scale, position};
    }
    FunctionBaseId base_id(size_t scale, size_t position) const {
        assert(scale < nb_scales);
        assert(position < power_of_2(scale));
        return power_of_2(scale) - 1 + position;
    }

    struct Wavelet {
        Interval<Bound::Open, Bound::Closed> up_part;
        Interval<Bound::Open, Bound::Closed> down_part;
        double normalization_factor;
    };
    Wavelet wavelet(size_t scale, size_t position) const {
        assert(scale < nb_scales);
        assert(position < power_of_2(scale));
        const double scale_factor = delta / double(power_of_2(scale + 1));
        const PointSpace left = (2 * position) * scale_factor;
        const PointSpace mid = (2 * position + 1) * scale_factor;
        const PointSpace right = (2 * position + 2) * scale_factor;
        const double normalization = delta_normalization_factor * std::pow(2, double(scale) / 2.);
        return {
            {left, mid},
            {mid, right},
            normalization,
        };
    }
    Wavelet wavelet(FunctionBaseId k) const {
        const ScalePosition sp = scale_and_position(k);
        return wavelet(sp.scale, sp.position);
    }

    std::string name() const final { return "haar wavelets base"; }
    void write_verbose_description(FILE * out) const final {
        fmt::print(
            out,
            "# base = Haar(nb_scales = {}, delta = {}) = {{\n"
            "#   w_{{s,p}} = sqrt(2)^s/sqrt(delta) * (\n"
            "#     Indicator_] delta * 2p / 2^(s+1), delta * (2p + 1) / 2^(s+1) ] -\n"
            "#     Indicator_] delta * (2p + 1) / 2^(s+1), delta * (2p + 2) / 2^(s+1) ]\n"
            "#   ) for s in [0, nb_scales[ ('scale'), p in [0, 2^s[ ('position')\n"
            "#   \n"
            "#   phi_k = w_{{s,p}} with k = 2^s - 1 + p <=> s = floor(log2(k)) and p = k + 1 - 2^s\n"
            "#   phi_k = {{w_{{0,0}}, w_{{1,0}}, w_{{1,1}}, w_{{2,0}}, w_{{2,1}}, w_{{2,2}}, w_{{2,3}}, ... }}\n"
            "# }}\n",
            nb_scales,
            delta);
    }
};

/******************************************************************************
 * Kernels.
 */

// Interval kernel : 1_[0, width](x) (L2-normalized)
struct IntervalKernel {
    PointSpace width; // ]0, inf[ due to the normalization factor

    IntervalKernel(PointSpace width_) : width(width_) { assert(width > 0.); }

    static std::string name() { return "interval[0,w]"; }
};
inline double normalization_factor(IntervalKernel kernel) {
    return 1. / double(kernel.width);
}

// Zero width kernels are not supported by computation, replace their width with a 'default' value.
inline PointSpace fix_zero_width(PointSpace width) {
    assert(width >= 0.);
    if(width == 0.) {
        return 1.;
    } else {
        return width;
    }
}

/* Kernel configuration is a combination of :
 * - kernel type (interval, etc)
 * - kernel granularity : no kernel, kernel by process (homogeneous), kernel by point (heterogeneous)
 *
 * Kernel configuration structures store kernel values for the selected granularity.
 * They all inherit KernelConfig for polymorphic storage ; it emulates a variant type.
 */
struct KernelConfig {
    virtual ~KernelConfig() = default;

    // User printable name
    virtual std::string name() const = 0;
    // Description generated in output file if verbose mode is used
    virtual void write_verbose_description(FILE * out) const = 0;
};

// No kernel, "point" mode.
struct NoKernel final : KernelConfig {
    std::string name() const final { return "no kernel"; }
    void write_verbose_description(FILE * out) const final { fmt::print(out, "# kernels = None\n"); }
};

// One kernel for each process.
// By default, chosen as the median of data_point interval widths.
template <typename KT> struct HomogeneousKernels final : KernelConfig {
    std::vector<KT> kernels; // For each process

    HomogeneousKernels(std::vector<KT> && kernels_) : kernels(std::move(kernels_)) {}

    std::string name() const final { return fmt::format("homogeneous {} kernels", KT::name()); }
    void write_verbose_description(FILE * out) const final {
        auto widths = map_to_vector(kernels, [](const KT & kernel) -> double { return kernel.width; });
        fmt::print(out, "# kernels = homogeneous {}, widths = {{{}}}\n", KT::name(), fmt::join(widths, ", "));
    }
};

// One kernel for each point in each process
template <typename KT> struct HeterogeneousKernels final : KernelConfig {
    DataByProcessRegion<std::vector<KT>> kernels; // With same order as points
    std::vector<KT> maximum_width_kernels;        // Maximum support of each process kernels (used for optimizations)

    HeterogeneousKernels(DataByProcessRegion<std::vector<KT>> && kernels_, std::vector<KT> && maximum_width_kernels_)
        : kernels(std::move(kernels_)), maximum_width_kernels(std::move(maximum_width_kernels_)) {
        assert(kernels.nb_processes() == maximum_width_kernels.size());
    }

    std::string name() const final { return fmt::format("heterogeneous {} kernels", KT::name()); }
    void write_verbose_description(FILE * out) const final {
        // Do not print widths, as they are too numerous
        fmt::print(out, "# kernels = heterogeneous {}\n", KT::name());
    }
};

/******************************************************************************
 * Computation matrices.
 */

/* Stores values for a_m_kl, b_m_kl, d_m_kl.
 *
 * Columns contains data for a process (m dimension).
 * Rows represent the {0}U{(l,k)} dimension, in the order: 0,(0,0),..,(0,K-1),(1,0),..,(1,K-1),...,(M-1,K-1).
 * Invariant: K > 0 && M > 0 && inner.size() == (1 + K * M, M).
 * Handles conversions of indexes to Eigen indexes (int).
 */
struct Matrix_M_MK1 {
    size_t nb_processes; // M
    size_t base_size;    // K
    Eigen::MatrixXd inner;

    Matrix_M_MK1(size_t nb_processes, size_t base_size) : nb_processes(nb_processes), base_size(base_size) {
        assert(nb_processes > 0);
        assert(base_size > 0);
        const auto size = 1 + base_size * nb_processes;
        inner = Eigen::MatrixXd::Constant(int(size), int(nb_processes), std::numeric_limits<double>::quiet_NaN());
    }

    // b_m,0
    double get_0(ProcessId m) const {
        assert(m < nb_processes);
        return inner(0, int(m));
    }
    void set_0(ProcessId m, double v) {
        assert(m < nb_processes);
        inner(0, int(m)) = v;
    }

    // b_m,l,k
    int lk_index(ProcessId l, FunctionBaseId k) const {
        assert(l < nb_processes);
        assert(k < base_size);
        return int(1 + l * base_size + k);
    }
    double get_lk(ProcessId m, ProcessId l, FunctionBaseId k) const {
        assert(m < nb_processes);
        return inner(lk_index(l, k), int(m));
    }
    void set_lk(ProcessId m, ProcessId l, FunctionBaseId k, double v) {
        assert(m < nb_processes);
        inner(lk_index(l, k), int(m)) = v;
    }

    // Access vector (m,0) for all m.
    auto m_0_values() const { return inner.row(0); }
    auto m_0_values() { return inner.row(0); }
    // Access sub-matrix (m,{(l,k)}) for all m,l,k
    auto m_lk_values() const { return inner.bottomRows(nb_processes * base_size); }
    auto m_lk_values() { return inner.bottomRows(nb_processes * base_size); }
    // Access vector of (m,{0}U{(l,k)}) for a given m.
    auto values_for_m(ProcessId m) const { return inner.col(m); }
    auto values_for_m(ProcessId m) { return inner.col(m); }
};

/* Stores value of the G matrix (symmetric).
 *
 * Rows and columns use the {0}U{(l,k)} dimension with the same order as Matrix_M_MK1.
 * Invariant: K > 0 && M > 0 && inner.size() == (1 + K * M, 1 + K * M)
 * Handles conversions of indexes to Eigen indexes (int).
 */
struct MatrixG {
    size_t nb_processes; // M
    size_t base_size;    // K
    Eigen::MatrixXd inner;

    MatrixG(size_t nb_processes, size_t base_size)
        : nb_processes(nb_processes),
          base_size(base_size),
          inner(1 + base_size * nb_processes, 1 + base_size * nb_processes) {
        assert(nb_processes > 0);
        assert(base_size > 0);
        const auto size = 1 + base_size * nb_processes;
        inner = Eigen::MatrixXd::Constant(int(size), size, std::numeric_limits<double>::quiet_NaN());
    }

    int lk_index(ProcessId l, FunctionBaseId k) const {
        assert(l < nb_processes);
        assert(k < base_size);
        return int(1 + l * base_size + k);
    }

    // Tmax
    double get_tmax() const { return inner(0, 0); }
    void set_tmax(double v) { inner(0, 0) = v; }

    // g_l,k (duplicated with transposition)
    double get_g(ProcessId l, FunctionBaseId k) const { return inner(0, lk_index(l, k)); }
    void set_g(ProcessId l, FunctionBaseId k, double v) {
        const auto i = lk_index(l, k);
        inner(0, i) = inner(i, 0) = v;
    }

    // G_l,l2_k,k2 (symmetric, only need to be set for (l,k) <= [or >=] (l2,k2))
    double get_G(ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) const {
        return inner(lk_index(l, k), lk_index(l2, k2));
    }
    void set_G(ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2, double v) {
        const auto i = lk_index(l, k);
        const auto i2 = lk_index(l2, k2);
        inner(i, i2) = inner(i2, i) = v;
    }
};
