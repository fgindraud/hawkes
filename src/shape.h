#pragma once

#include <algorithm> // add
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "types.h"

/******************************************************************************
 * Shape definitions, manipulation, and computations on shapes.
 *
 * A shape is a function f : R -> R with a finite support (non zero domain).
 * The non zero domain of f is an interval of R where f may be non null.
 * For all x in R - nzd(f), f(x) = 0.
 *
 * Shapes are represented by C++ structs with the given methods :
 * - double operator()(Point x) const : computes f(x) for any x in R.
 * - Interval<?, ?> non_zero_domain() const : returns the non_zero_domain.
 *
 * With these definitions, convolution and other similar operations can be defined as:
 * convolution(ShapeA, ShapeB) -> ShapeC
 * The type of ShapeC depends on ShapeA & ShapeB, the choice is done by overloading.
 */
namespace shape {

using ::Bound;
using ::Interval;
using ::Point;
using ::PointSpace;

// Get non_zero_domain interval type from a shape
template <typename Shape> using NzdIntervalType = decltype(std::declval<Shape>().non_zero_domain());

#if 0
inline double square(double x) {
    return x * x;
}
inline double cube(double x) {
    return x * square(x);
}

/* Triangle (0,0), (side, 0), (side, side).
 */
struct PositiveTriangle {
    PointSpace side; // [0, inf[

    PositiveTriangle(PointSpace side) : side(side) { assert(side >= 0.); }

    ClosedInterval non_zero_domain() const { return {0., side}; }
    double operator()(Point x) const {
        if(non_zero_domain().contains(x)) {
            return x;
        } else {
            return 0.;
        }
    }
};

/* Triangle (0,0), (-side, 0), (-side, side).
 */
struct NegativeTriangle {
    PointSpace side; // [0, inf[

    NegativeTriangle(PointSpace side) : side(side) { assert(side >= 0.); }

    ClosedInterval non_zero_domain() const { return {-side, 0.}; }
    double operator()(Point x) const {
        if(non_zero_domain().contains(x)) {
            return -x;
        } else {
            return 0.;
        }
    }
};
inline auto as_positive_triangle(NegativeTriangle t) {
    // NegativeTriangle(side)(x) == PositiveTriangle(side)(x)
    return reversed(PositiveTriangle{t.side});
}

/* Trapezoid with a block (2*half_base, height) with a PositiveTriangle(height) on the left and a negative on the right.
 */
struct Trapezoid {
    PointSpace height;    // [0, inf[
    PointSpace half_base; // [0, inf[
    PointSpace half_len;  // Precomputed

    Trapezoid(PointSpace height, PointSpace half_base) : height(height), half_base(half_base) {
        assert(height >= 0.);
        assert(half_base >= 0.);
        half_len = half_base + height;
    }

    ClosedInterval non_zero_domain() const { return {-half_len, half_len}; }
    double operator()(Point x) const {
        if(!non_zero_domain().contains(x)) {
            return 0.;
        } else if(x < -half_base) {
            return x + half_len; // Left triangle
        } else if(x > half_base) {
            return half_len - x; // Right triangle
        } else {
            return height; // Central block
        }
    }
};

/* Convolution between IntervalIndicator(half_width=l/2) and PositiveTriangle(side=c).
 */
struct ConvolutionIntervalPositiveTriangle {
    PointSpace half_l; // [0, inf[
    PointSpace c;      // [0, inf[
    // Precomputed values
    PointSpace central_section_left;
    PointSpace central_section_right;

    ConvolutionIntervalPositiveTriangle(PointSpace half_l, PointSpace c) : half_l(half_l), c(c) {
        assert(half_l >= 0.);
        assert(c >= 0.);
        std::tie(central_section_left, central_section_right) = std::minmax(half_l, c - half_l);
        // Check bounds
        assert(-half_l <= central_section_left);
        assert(central_section_left <= central_section_right);
        assert(central_section_right <= c + half_l);
    }

    ClosedInterval non_zero_domain() const { return {-half_l, c + half_l}; }
    double operator()(Point x) const {
        if(!non_zero_domain().contains(x)) {
            return 0.;
        } else if(x < central_section_left) {
            return square(x + half_l) / 2.; // Quadratic left part
        } else if(x > central_section_right) {
            return (square(c) - square(x - half_l)) / 2.; // Quadratic right part
        } else {
            // Central section has two behaviors depending on l <=> c
            if(2. * half_l >= c) {
                return square(c) / 2.; // l >= c : constant central part
            } else {
                return 2. * half_l * x; // l < c : linear central part
            }
        }
    }
};

/* Convolution between PositiveTriangle(side=a) and PositiveTriangle(side=b).
 */
struct ConvolutionPositiveTrianglePositiveTriangle {
    // Precomputed values (definitions in the shape doc)
    PointSpace a_plus_b;
    PointSpace A;
    PointSpace B;
    double polynom_constant;

    ConvolutionPositiveTrianglePositiveTriangle(PointSpace a, PointSpace b) {
        assert(a >= 0.);
        assert(b >= 0.);
        a_plus_b = a + b;
        std::tie(A, B) = std::minmax(a, b);
        // Polynom constant used for cubic right part = -2a^2 -2b^2 + 2ab.
        polynom_constant = 2. * (3. * a * b - square(a_plus_b));
        // Check bounds
        assert(0. <= A);
        assert(A <= B);
        assert(B <= a_plus_b);
    }

    ClosedInterval non_zero_domain() const { return {0., a_plus_b}; }
    double operator()(Point x) const {
        if(!non_zero_domain().contains(x)) {
            return 0.;
        } else if(x < A) {
            return cube(x) / 6.; // Cubic left part
        } else if(x > B) {
            // Cubic right part
            const auto polynom = x * (x + a_plus_b) + polynom_constant;
            return (a_plus_b - x) * polynom / 6.;
        } else {
            return square(A) * (3. * x - 2. * A) / 6.; // Central section has one formula using A=min(a,b)
        }
    }
};

/* Convolution between NegativeTriangle(side=a) and PositiveTriangle(side=b).
 */
struct ConvolutionNegativeTrianglePositiveTriangle {
    PointSpace a;
    PointSpace b;
    // Precomputed values
    PointSpace A;
    PointSpace B;

    ConvolutionNegativeTrianglePositiveTriangle(PointSpace a, PointSpace b) : a(a), b(b) {
        assert(a >= 0.);
        assert(b >= 0.);
        std::tie(A, B) = std::minmax(0., b - a);
        // Check bounds
        assert(-a <= A);
        assert(A <= B);
        assert(B <= b);
    }

    ClosedInterval non_zero_domain() const { return {-a, b}; }
    double operator()(Point x) const {
        if(!non_zero_domain().contains(x)) {
            return 0.;
        } else if(x < A) {
            return square(x + a) * (2. * a - x) / 6.; // Cubic left part
        } else if(x > B) {
            return square(b - x) * (2. * b + x) / 6.; // Cubic right part
        } else {
            // Central section has two behaviors depending on a <=> b
            if(a < b) {
                return square(a) * (2. * a + 3. * x) / 6.; // Linear central part for [0, b-a].
            } else {
                return square(b) * (2. * b - 3. * x) / 6.; // Linear central part for [b-a, 0].
            }
        }
    }
};
#endif

/******************************************************************************
 * Base shapes.
 */

/* Indicator function for an interval.
 * Returns 1 if x in interval, 0 if outside.
 * non_zero_domain is exactly the interval.
 */
template <Bound lb, Bound rb> struct Indicator {
    Interval<lb, rb> interval;

    Interval<lb, rb> non_zero_domain() const { return interval; }
    double operator()(Point x) const {
        if(interval.contains(x)) {
            return 1.;
        } else {
            return 0.;
        }
    }
};

/* Polynom on a restricted interval.
 * f(x) = if(x in interval) { sum_i coefficients[i] x^i } else { 0 }
 */
inline double compute_polynom_value(Point x, span<const double> coefficients) {
    assert(coefficients.size() > 0);
    // Horner strategy
    size_t i = coefficients.size() - 1;
    double r = coefficients[i];
    while(i > 0) {
        i -= 1;
        r = coefficients[i] + x * r;
    }
    return r;
}
template <Bound lb, Bound rb> struct Polynom {
    Interval<lb, rb> interval;
    std::vector<double> coefficients; // size() > 0

    // With zeroed coefficients
    Polynom(Interval<lb, rb> interval_, size_t degree) : interval(interval_), coefficients(degree + 1, 0.) {}
    // With values
    Polynom(Interval<lb, rb> interval_, std::vector<double> && coefficients_)
        : interval(interval_), coefficients(std::move(coefficients_)) {
        assert(coefficients.size() > 0);
    }

    size_t degree() const { return coefficients.size() - 1; }

    Interval<lb, rb> non_zero_domain() const { return interval; }
    double operator()(Point x) const {
        if(interval.contains(x)) {
            return compute_polynom_value(x, make_span(coefficients));
        } else {
            return 0.;
        }
    }
};

/******************************************************************************
 * Shape manipulation functions and combinators.
 * Combinators : modify a shape, like x/y translation, x/y scaling, ...
 *
 * Strategy :
 * Define explicit cases for base shapes.
 * Define simplifications when combinators are found.
 *
 * The conventional order of combinators is:
 * scaling -> add -> base_shape
 */

/* Reverse x dimension
 * reverse(f)(x) = f(-x)
 */
template <Bound lb, Bound rb> Indicator<rb, lb> reverse(const Indicator<lb, rb> & indicator) {
    return {-indicator.interval};
}
template <Bound lb, Bound rb> Polynom<rb, lb> reverse(const Polynom<lb, rb> & polynom) {
    // Q(x) = P(-x) = sum_k a_k (-x)^k = sum_k (a_k (-1)^k) x^k for -x in nzd(P) <=> x in -nzd(P).
    // Copy and invert coefficients for odd k.
    auto coefficients = std::vector<double>(polynom.coefficients);
    for(size_t k = 1; k < coefficients.size(); k += 2) {
        coefficients[k] = -coefficients[k];
    }
    return {-polynom.interval, std::move(coefficients)};
}

/* Shift x dimension
 * shifted(s, f)(x) = f(x - s)
 *
 * Defined entirely through overloads on specific shape classes.
 * All base shapes have a builtin shifting (interval).
 */
template <Bound lb, Bound rb> Indicator<rb, lb> shifted(PointSpace s, const Indicator<lb, rb> & ind) {
    return {s + ind.interval};
}

/* Scale y dimension
 * scaled(s, f)(x) = s * f(x)
 */
template <typename Inner> struct Scaled {
    double scale;
    Inner inner;

    NzdIntervalType<Inner> non_zero_domain() const { return inner.non_zero_domain(); }
    double operator()(Point x) const { return scale * inner(x); }
};
template <typename Inner> inline auto scaled(double scale, const Inner & inner) {
    return Scaled<Inner>{scale, inner};
}

template <typename Inner> inline auto scaled(double scale, const Scaled<Inner> & s) {
    return scaled(scale * s.scale, s.inner);
}

/* Addition / composition of multiple shapes.
 */
template <typename Container> struct Add;

// Vector of shapes of uniform type. Performs optimisations like dropping zero shapes.
template <typename Inner> struct Add<std::vector<Inner>> {
    std::vector<Inner> components;
    NzdIntervalType<Inner> union_non_zero_domain{0., 0.};

    Add() = default; // Zero function

    Add(std::vector<Inner> && components_) : components(std::move(components_)) {
        if(NzdIntervalType<Inner>::left_bound_type == Bound::Open ||
           NzdIntervalType<Inner>::right_bound_type == Bound::Open) {
            // Remove zero width components, except if both bounds are closed (it changes the overall value).
            auto new_end = std::remove_if(components.begin(), components.end(), [](const Inner & shape) {
                return shape.non_zero_domain().width() == 0.;
            });
            components.erase(new_end, components.end());
        }
        // TODO sum polynoms could merge polynoms with same support.
        if(!components.empty()) {
            // Determine the non zero domain
            union_non_zero_domain = components[0].non_zero_domain();
            for(size_t i = 1; i < components.size(); ++i) {
                union_non_zero_domain = union_(union_non_zero_domain, components[i].non_zero_domain());
            }
        }
    }

    NzdIntervalType<Inner> non_zero_domain() const { return union_non_zero_domain; }
    double operator()(Point x) const {
        if(union_non_zero_domain.contains(x)) {
            double sum = 0.;
            for(const Inner & component : components) {
                sum += component(x);
            }
            return sum;
        } else {
            return 0.;
        }
    }
};

/******************************************************************************
 * Computation tools for convolution / cross_correlation of polynomials.
 *
 * The convolution of two polynom shapes is a composition of 3 polynom shapes.
 * This allows one implementation to compute convolutions for an entire family of shapes.
 * The convolution does not care about the type of bounds of input polynoms, so they are ignored.
 *
 * Similar properties for the cross_correlation.
 */

constexpr double indicator_polynomial_coefficients[1] = {1.};

// Temporary reference to a shape similar to a polynom without bound types.
// Supports implicit conversion from valid shapes.
struct Polynomial {
    Point interval_left;
    Point interval_right;
    span<const double> coefficients;

    template <Bound lb, Bound rb> Polynomial(const Indicator<lb, rb> & indicator)
        : interval_left(indicator.interval.left),
          interval_right(indicator.interval.right),
          coefficients(make_span(indicator_polynomial_coefficients)) {}

    template <Bound lb, Bound rb> Polynomial(const Polynom<lb, rb> & polynom)
        : interval_left(polynom.interval.left),
          interval_right(polynom.interval.right),
          coefficients(make_span(polynom.coefficients)) {}

    size_t degree() const { return coefficients.size() - 1; }
    PointSpace width() const { return interval_right - interval_left; }
};

struct BinomialCoefficientsUpToN {
    std::vector<std::vector<size_t>> values;

    explicit BinomialCoefficientsUpToN(size_t maximum_n) {
        values.reserve(maximum_n + 1);
        for(size_t n = 0; n <= maximum_n; ++n) {
            std::vector<size_t> values_for_n(n + 1);
            values_for_n[0] = 1;
            values_for_n[n] = 1;
            for(size_t k = 1; k + 1 <= n; ++k) {
                // k <= n - 1 generates wrapping underflow for n = 0 !
                values_for_n[k] = values[n - 1][k - 1] + values[n - 1][k];
            }
            values.emplace_back(std::move(values_for_n));
        }
    }

    size_t maximum_n() const { return values.size() - 1; }
    size_t operator()(size_t k, size_t n) const {
        assert(k <= n);
        assert(n <= maximum_n());
        return values[n][k];
    }
};

struct PowersUpToN {
    std::vector<double> values;

    PowersUpToN(size_t maximum_n, double c) : values(maximum_n + 1) {
        values[0] = 1.;
        for(size_t n = 1; n <= maximum_n; ++n) {
            values[n] = values[n - 1] * c;
        }
    }

    size_t maximum_n() const { return values.size() - 1; }
    double operator()(size_t n) const {
        assert(n <= maximum_n());
        return values[n];
    }
};

/******************************************************************************
 * Convolution.
 *
 * convolution(f,g)(x) = int_R f(x - t) g (t) dt
 *
 * Perform simplification by factoring scaling and shifting before applying base shape convolution.
 */
template <typename Lhs, typename Rhs> inline auto convolution(const Lhs & lhs, const Rhs & rhs) {
    return convolution_extract_scale(lhs, rhs);
}

// Extract scale
template <typename L, typename R> inline auto convolution_extract_scale(const Scaled<L> & lhs, const Scaled<R> & rhs) {
    return scaled(lhs.scale * rhs.scale, convolution_base(lhs.inner, rhs.inner));
}
template <typename L, typename R> inline auto convolution_extract_scale(const Scaled<L> & lhs, const R & rhs) {
    return scaled(lhs.scale, convolution_base(lhs.inner, rhs));
}
template <typename L, typename R> inline auto convolution_extract_scale(const L & lhs, const Scaled<R> & rhs) {
    return scaled(rhs.scale, convolution_base(lhs, rhs.inner));
}
template <typename L, typename R> inline auto convolution_extract_scale(const L & lhs, const R & rhs) {
    return convolution_base(lhs, rhs);
}

/* Computes the three polynomial components of a convolution.
 *
 * Components are defined on consecutive intervals: ]0,qw] ]qw,pw] ]pw,pw+qw].
 * Open-Closed bounds are chosen so the sum contains 1 value at points {qw, pw} (borders).
 * Open-Closed is chosen instead of Closed-Open to better fit hawkes computation (cases with ]0,?]).
 * The Open bound on ]0,qw] is ok, as left_part(0) == a_0 == 0 by construction of the formulas.
 *
 * Testable properties:
 * Matching values at component borders
 * 0s at points {0, qw + pw} ; the convolution on the sides of lhs/rhs nzds.
 * Explicitly known coefficients in fixed cases: trapezoid, etc...
 */
inline void append_convolution_components(
    std::vector<Polynom<Bound::Open, Bound::Closed>> & components, Polynomial lhs, Polynomial rhs) {
    // q is the polynom with the smallest width
    const auto compare_widths = [](const auto & lhs, const auto & rhs) { return lhs.width() < rhs.width(); };
    const Polynomial & p = std::max(lhs, rhs, compare_widths);
    const Polynomial & q = std::min(lhs, rhs, compare_widths);
    if(q.width() == 0.) {
        return; // Optimization : convolution with zero width support is a zero value.
    }
    // Useful numerical tools / values
    const size_t border_parts_degree = p.degree() + q.degree() + 1;
    const size_t center_part_degree = p.degree();
    const auto binomial = BinomialCoefficientsUpToN(border_parts_degree);
    const auto ipl_powers = PowersUpToN(border_parts_degree, p.interval_left);
    const auto ipr_powers = PowersUpToN(border_parts_degree, p.interval_right);
    const auto iql_powers = PowersUpToN(border_parts_degree, q.interval_left);
    const auto iqr_powers = PowersUpToN(border_parts_degree, q.interval_right);
    const auto minus_1_power = [](size_t n) -> double { return n % 2 == 0 ? 1. : -1.; };
    // Left part
    {
        auto coefficients = std::vector<double>(border_parts_degree + 1, 0.);
        for(size_t k = 0; k <= p.degree(); ++k) {
            for(size_t j = 0; j <= q.degree(); ++j) {
                for(size_t i = 0; i <= k; ++i) {
                    const size_t ij1 = i + j + 1;
                    const double factor =
                        p.coefficients[k] * q.coefficients[j] * binomial(i, k) * minus_1_power(i) / double(ij1);
                    coefficients[k - i] += -factor * iql_powers(ij1);
                    for(size_t l = 0; l <= ij1; ++l) {
                        coefficients[k + j + 1 - l] += factor * binomial(l, ij1) * minus_1_power(l) * ipl_powers(l);
                    }
                }
            }
        }
        components.emplace_back(
            Interval<Bound::Open, Bound::Closed>{
                p.interval_left + q.interval_left,
                p.interval_left + q.interval_right,
            },
            std::move(coefficients));
    }
    // Center part. Optimization : avoid computing it if zero width (would be discarded).
    if(p.width() > q.width()) {
        auto coefficients = std::vector<double>(center_part_degree + 1, 0.);
        for(size_t k = 0; k <= p.degree(); ++k) {
            for(size_t j = 0; j <= q.degree(); ++j) {
                for(size_t i = 0; i <= k; ++i) {
                    const size_t ij1 = i + j + 1;
                    coefficients[k - i] += p.coefficients[k] * q.coefficients[j] * binomial(i, k) *
                                           (minus_1_power(i) / double(ij1)) * (iqr_powers(ij1) - iql_powers(ij1));
                }
            }
        }
        components.emplace_back(
            Interval<Bound::Open, Bound::Closed>{
                p.interval_left + q.interval_right,
                p.interval_right + q.interval_left,
            },
            std::move(coefficients));
    }
    // Right part
    {
        auto coefficients = std::vector<double>(border_parts_degree + 1, 0.);
        for(size_t k = 0; k <= p.degree(); ++k) {
            for(size_t j = 0; j <= q.degree(); ++j) {
                for(size_t i = 0; i <= k; ++i) {
                    const size_t ij1 = i + j + 1;
                    const double factor =
                        p.coefficients[k] * q.coefficients[j] * binomial(i, k) * minus_1_power(i) / double(ij1);
                    coefficients[k - i] += factor * iqr_powers(ij1);
                    for(size_t l = 0; l <= ij1; ++l) {
                        coefficients[k + j + 1 - l] += -factor * binomial(l, ij1) * minus_1_power(l) * ipr_powers(l);
                    }
                }
            }
        }
        components.emplace_back(
            Interval<Bound::Open, Bound::Closed>{
                p.interval_right + q.interval_left,
                p.interval_right + q.interval_right,
            },
            std::move(coefficients));
    }
}

// Base cases, including distributed convolution for Add<Polynom>>.
inline Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> convolution_base(Polynomial lhs, Polynomial rhs) {
    std::vector<Polynom<Bound::Open, Bound::Closed>> components;
    append_convolution_components(components, lhs, rhs);
    return {std::move(components)};
}

inline Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> convolution_base(
    const Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> & lhs, Polynomial rhs) {
    std::vector<Polynom<Bound::Open, Bound::Closed>> components;
    for(const auto & lhs_component : lhs.components) {
        append_convolution_components(components, lhs_component, rhs);
    }
    return {std::move(components)};
}
inline Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> convolution_base(
    Polynomial lhs, const Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> & rhs) {
    return convolution_base(rhs, lhs); // Use commutativity.
}

inline Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> convolution_base(
    const Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> & lhs,
    const Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> & rhs) {
    std::vector<Polynom<Bound::Open, Bound::Closed>> components;
    for(const auto & lhs_component : lhs.components) {
        for(const auto & rhs_component : rhs.components) {
            append_convolution_components(components, lhs_component, rhs_component);
        }
    }
    return {std::move(components)};
}

/******************************************************************************
 * Cross correlation.
 *
 * cross_correlation(f,g)(x) = int_R f(t - x) g (t) dt = convolution(reverse(f), g).
 *
 * Perform simplification by factoring scaling and shifting before applying base shape convolution.
 * Often use definitions from convolution.
 */
template <typename Lhs, typename Rhs> inline auto cross_correlation(const Lhs & lhs, const Rhs & rhs) {
    return cross_correlation_extract_scale(lhs, rhs);
}

// Extract scale
template <typename L, typename R>
inline auto cross_correlation_extract_scale(const Scaled<L> & lhs, const Scaled<R> & rhs) {
    return scaled(lhs.scale * rhs.scale, cross_correlation_base(lhs.inner, rhs.inner));
}
template <typename L, typename R> inline auto cross_correlation_extract_scale(const Scaled<L> & lhs, const R & rhs) {
    return scaled(lhs.scale, cross_correlation_base(lhs.inner, rhs));
}
template <typename L, typename R> inline auto cross_correlation_extract_scale(const L & lhs, const Scaled<R> & rhs) {
    return scaled(rhs.scale, cross_correlation_base(lhs, rhs.inner));
}
template <typename L, typename R> inline auto cross_correlation_extract_scale(const L & lhs, const R & rhs) {
    return cross_correlation_base(lhs, rhs);
}

/* Base cases, including distributed cross_correlation for Add<Polynom>>.

 * Currently uses convolution with reverse(p) for simplicity.
 * reverse(p) does not work for Polynomial, so duplicate for individual cases.
 * TODO better scheme ? compute formulas for cross_correlation ?
 */
template <Bound lb, Bound rb> inline Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> cross_correlation_base(
    Indicator<lb, rb> lhs, Polynomial rhs) {
    std::vector<Polynom<Bound::Open, Bound::Closed>> components;
    append_convolution_components(components, reverse(lhs), rhs);
    return {std::move(components)};
}
template <Bound lb, Bound rb> inline Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> cross_correlation_base(
    const Polynom<lb, rb> & lhs, Polynomial rhs) {
    std::vector<Polynom<Bound::Open, Bound::Closed>> components;
    append_convolution_components(components, reverse(lhs), rhs);
    return {std::move(components)};
}

inline Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> cross_correlation_base(
    const Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> & lhs, Polynomial rhs) {
    std::vector<Polynom<Bound::Open, Bound::Closed>> components;
    for(const auto & lhs_component : lhs.components) {
        append_convolution_components(components, reverse(lhs_component), rhs);
    }
    return {std::move(components)};
}

template <Bound lb, Bound rb> inline Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> cross_correlation_base(
    Indicator<lb, rb> lhs, const Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> & rhs) {
    std::vector<Polynom<Bound::Open, Bound::Closed>> components;
    const auto reverse_lhs = reverse(lhs);
    for(const auto & rhs_component : rhs.components) {
        append_convolution_components(components, reverse_lhs, rhs_component);
    }
    return {std::move(components)};
}
template <Bound lb, Bound rb> inline Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> cross_correlation_base(
    const Polynom<lb, rb> & lhs, const Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> & rhs) {
    std::vector<Polynom<Bound::Open, Bound::Closed>> components;
    const auto reverse_lhs = reverse(lhs);
    for(const auto & rhs_component : rhs.components) {
        append_convolution_components(components, reverse_lhs, rhs_component);
    }
    return {std::move(components)};
}

inline Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> cross_correlation_base(
    const Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> & lhs,
    const Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> & rhs) {
    std::vector<Polynom<Bound::Open, Bound::Closed>> components;
    for(const auto & lhs_component : lhs.components) {
        const auto reverse_lhs_component = reverse(lhs_component);
        for(const auto & rhs_component : rhs.components) {
            append_convolution_components(components, reverse_lhs_component, rhs_component);
        }
    }
    return {std::move(components)};
}

/******************************************************************************
 * Approximate a shape with an interval.
 *
 * For polynoms (or sums of polynoms), use the average value spread over the definition domain.
 */
template <typename Inner> inline auto indicator_approximation(const Scaled<Inner> & shape) {
    return scaled(shape.scale, indicator_approximation(shape.inner));
}

template <Bound lb, Bound rb> inline Indicator<lb, rb> indicator_approximation(const Indicator<lb, rb> & indicator) {
    return indicator;
}

// Compute integral of polynomial or sums of polynomials
inline double integral(Polynomial polynomial) {
    // int_R sum_i a_i x^i = int_[0,w] sum_i a_i x^i = sum_i a_i w^(i+1) / (i+1)
    // Shift is ignored.
    const double width = polynomial.width();
    const span<const double> coefficients = polynomial.coefficients;
    assert(coefficients.size() > 0);
    // Horner strategy : sum_i a_i w^i / (i+1)
    size_t i = coefficients.size() - 1;
    double r = coefficients[i] / double(i + 1);
    while(i > 0) {
        i -= 1;
        r = coefficients[i] / double(i + 1) + width * r;
    }
    // Add final width multiplier
    return r * width;
}
template <typename T> inline double integral(const Add<std::vector<T>> & sum) {
    double r = 0;
    for(const T & component : sum.components) {
        r += integral(component);
    }
    return r;
}

template <Bound lb, Bound rb>
inline Scaled<Indicator<lb, rb>> indicator_approximation(const Polynom<lb, rb> & polynom) {
    const Interval<lb, rb> non_zero_domain = polynom.non_zero_domain();
    const PointSpace width = non_zero_domain.width();
    if(width == 0.) {
        return Scaled<Indicator<lb, rb>>{
            0.,
            {{0., 0.}},
        }; // Dummy 0 value
    } else {
        const double average_value = integral(polynom) / width;
        return Scaled<Indicator<lb, rb>>{
            average_value,
            {non_zero_domain},
        };
    }
}

template <Bound lb, Bound rb>
inline Scaled<Indicator<lb, rb>> indicator_approximation(const Add<std::vector<Polynom<lb, rb>>> & polynom_sum) {
    const Interval<lb, rb> non_zero_domain = polynom_sum.non_zero_domain();
    const PointSpace width = non_zero_domain.width();
    if(width == 0.) {
        return Scaled<Indicator<lb, rb>>{
            0.,
            {{0., 0.}},
        }; // Dummy 0 value
    } else {
        const double average_value = integral(polynom_sum) / width;
        return Scaled<Indicator<lb, rb>>{
            average_value,
            {non_zero_domain},
        };
    }
}

/******************************************************************************
 * Computations with shapes.
 */

/* ShiftedPoints iteration tool.
 * This is an iterator over the coordinates of points, all shifted by the given shift.
 *
 * Typical usage:
 * Have multiple instances with various shifts to represent sets of interesting Point positions.
 * At each step, find the next interesting point by taking the minimum of point() values.
 * Do computation for the point.
 * Advance all iterators using next_if_equal : iterators which were at this exact point will advance.
 */
class ShiftedPoints {
  public:
    static constexpr PointSpace inf = std::numeric_limits<PointSpace>::infinity();

    ShiftedPoints(const SortedVec<Point> & points, PointSpace shift) : points_(points), shift_(shift) {
        set_current(0);
    }

    size_t index() const { return index_; }
    Point point() const { return shifted_point_; }

    void next_if_equal(Point processed_point) {
        if(processed_point == shifted_point_) {
            set_current(index_ + 1);
        }
    }

  private:
    const SortedVec<Point> & points_; // Points to iterate on
    PointSpace shift_;                // Shifting from points

    // Iterator current index and shifted point value (or inf if out of points)
    size_t index_;
    Point shifted_point_;

    void set_current(size_t i) {
        index_ = i;
        shifted_point_ = i < points_.size() ? points_[i] + shift_ : inf;
    }
};

/* Compute sum_{x_m in N_m, x_l in N_l} shape(x_m - x_l).
 * Shape must be a valid shape as per the shape namespace definition:
 * - has a method non_zero_domain() returning the Interval where the shape is non zero.
 * - has an operator()(x) returning the value at point x.
 *
 * Worst case complexity: O(|N|^2).
 * Average complexity: O(|N| * density(N) * width(shape)) = O(|N|^2 * width(shape) / Tmax).
 */
template <typename Shape> inline double sum_of_point_differences(
    const SortedVec<Point> & m_points, const SortedVec<Point> & l_points, const Shape & shape) {
    // shape(x) != 0 => x in shape.non_zero_domain().
    // Thus sum_{x_m,x_l} shape(x_m - x_l) = sum_{(x_m, x_l), x_m - x_l in non_zero_domain} shape(x_m - x_l).
    const auto non_zero_domain = shape.non_zero_domain();

    double sum = 0.;
    size_t starting_i_m = 0;
    for(const Point x_l : l_points) {
        // x_l = N_l[i_l], with N_l[x] a strictly increasing function of x.
        // Compute shape(x_m - x_l) for all x_m in (x_l + non_zero_domain) interval.
        const auto interval_i_l = x_l + non_zero_domain;

        // starting_i_m = min{i_m, N_m[i_m] - N_l[i_l] >= non_zero_domain.left}.
        // We can restrict the search by starting from:
        // last_starting_i_m = min{i_m, N_m[i_m] - N_l[i_l - 1] >= non_zero_domain.left or i_m == 0}.
        // We have: N_m[starting_i_m] >= N_l[i_l] + nzd.left > N_l[i_l - 1] + nzd.left.
        // Because N_m is increasing and properties of the min, starting_i_m >= last_starting_i_m.
        while(starting_i_m < m_points.size() && !(interval_i_l.in_left_bound(m_points[starting_i_m]))) {
            starting_i_m += 1;
        }
        if(starting_i_m == m_points.size()) {
            // starting_i_m is undefined because last(N_m) < N_l[i_l] + non_zero_domain.left.
            // last(N_m) == max(x_m in N_m) because N_m[x] is strictly increasing.
            // So for each j > i_l , max(x_m) < N[j] + non_zero_domain.left, and shape (x_m - N_l[j]) == 0.
            // We can stop there as the sum is already complete.
            break;
        }
        // Sum values of shape(x_m - x_l) as long as x_m is in interval_i_l.
        // starting_i_m defined => for each i_m < starting_i_m, shape(N_m[i_m] - x_l) == 0.
        // Thus we only scan from starting_i_m to the last i_m in interval.
        // N_m[x] is strictly increasing so we only need to check the right bound of the interval.
        for(size_t i_m = starting_i_m; i_m < m_points.size() && interval_i_l.in_right_bound(m_points[i_m]); i_m += 1) {
            sum += shape(m_points[i_m] - x_l);
        }
    }
    return sum;
}

// TODO version specific to Indicator, with linear speed

// Scaling can be moved out of computation.
template <typename Inner> inline double sum_of_point_differences(
    const SortedVec<Point> & m_points, const SortedVec<Point> & l_points, const shape::Scaled<Inner> & shape) {
    return shape.scale * sum_of_point_differences(m_points, l_points, shape.inner);
}

/* Compute sup_{x} sum_{y in points} shape(x - y).
 * This is a building block for computation of B_hat, used in the computation of lasso penalties (d).
 */
template <Bound lb, Bound rb> inline double sup_of_sum_of_differences_to_points(
    const SortedVec<Point> & points, const Indicator<lb, rb> & indicator) {
    /* For an indicator function, the sum is a piecewise constant function of x.
     * This function has at maximum 2*|N_l| points of change, so the number of different values is finite.
     * Thus the sup over x is a max over all these possible values.
     *
     * Assuming a closed interval (but similar for other bound types):
     * x in interval for y <=> left <= x - y <= right <=> left + y <= x <= right + x.
     * Thus we iterate on the sets of points {left + y} and {right + y} where the sum changes of value (+1 and -1).
     */
    ShiftedPoints left_bounds(points, indicator.interval.left);
    ShiftedPoints right_bounds(points, indicator.interval.right);
    double max = 0;
    while(true) {
        // x is the next left / right / both interval bound.
        const Point x = std::min(left_bounds.point(), right_bounds.point());
        if(x == ShiftedPoints::inf) {
            break; // No more points to process.
        }
        // The sum of indicator at x is the number of entered intervals minus the number of exited intervals.
        // Thus the sum is the difference between the indexes of the left bound iterator and the right one.
        if(lb == Bound::Closed && rb == Bound::Closed) {
            // If both bounds are Closed, we look at exactly x, iterating the left points before looking.
            // sum(x) >= sum(x-) as we could overlap an entering and exiting interval.
            left_bounds.next_if_equal(x);
        }
        assert(left_bounds.index() >= right_bounds.index());
        const auto sum_value_for_x = PointSpace(left_bounds.index() - right_bounds.index());
        max = std::max(max, sum_value_for_x);
        if(!(lb == Bound::Closed && rb == Bound::Closed)) {
            // If any of the bounds is Open, we look at x- (before iterating after x).
            // sum(x-) >= sum(x) due to open bounds.
            left_bounds.next_if_equal(x);
        }
        right_bounds.next_if_equal(x);
    }
    return max;
}

// Scaling can be moved out
template <typename Inner>
inline double sup_of_sum_of_differences_to_points(const SortedVec<Point> & points, const Scaled<Inner> & shape) {
    if(shape.scale > 0.) {
        return shape.scale * sup_of_sum_of_differences_to_points(points, shape.inner);
    } else {
        return 0.;
    }
}

} // namespace shape
