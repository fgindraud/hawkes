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
 * interval = [center - half_width, center + half_width]
 * P(x) = if(x in interval) { sum_i coefficients[i] (x - interval.center)^i } else { 0 }
 *
 * A previous definition was P(x) = sum_i a_k x^k for x in interval.
 * For an interval far from 0, and P shape "small" : max{|P(x)| for x in interval} << |interval.center|.
 * a_0 would need to be huge to compensate x^N_p at the interval level and ensure a "small" shape.
 * This means generating a "small" difference from multiple huge floating point numbers.
 * This is usually very imprecise. Tests for continuity of convolution failed due to this.
 *
 * To reduce this imprecision, the "origin" of the polynom is fixed at the interval center.
 * Thus the above expression. This is equivalent to a 0-centered polynom, shifted to interval center.
 */
inline double compute_polynom_value(double x, span<const double> coefficients) {
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

    Polynom(Interval<lb, rb> interval_, std::vector<double> && coefficients_)
        : interval(interval_), coefficients(std::move(coefficients_)) {
        assert(coefficients.size() > 0);
    }

    size_t degree() const { return coefficients.size() - 1; }

    Interval<lb, rb> non_zero_domain() const { return interval; }
    double operator()(Point x) const {
        if(interval.contains(x)) {
            return compute_polynom_value(x - interval.center(), make_span(coefficients));
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
    // Q(x) = P(-x) = sum_k a_k (-x - origin)^k = sum_k (a_k (-1)^k) (x - -origin)^k for -x in nzd(P) <=> x in -nzd(P).
    // Copy and invert coefficients for odd k.
    auto coefficients = std::vector<double>(polynom.coefficients);
    for(size_t k = 1; k < coefficients.size(); k += 2) {
        coefficients[k] = -coefficients[k];
    }
    // origin = center(nzd(P)) <=> -origin = center(-nzd(P)), deduced from interval
    return {-polynom.interval, std::move(coefficients)};
}

/* Shift x dimension
 * shifted(s, f)(x) = f(x - s)
 *
 * Defined entirely through overloads on specific shape classes.
 * All base shapes have a builtin shifting (interval).
 */
template <Bound lb, Bound rb> Indicator<lb, rb> shifted(PointSpace s, const Indicator<lb, rb> & ind) {
    return {s + ind.interval};
}
template <Bound lb, Bound rb> Polynom<lb, rb> shifted(PointSpace s, const Polynom<lb, rb> & p) {
    return {s + p.interval, std::vector<double>{p.coefficients}};
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
    Point origin;
    PointSpace half_width;
    span<const double> coefficients;

    template <Bound lb, Bound rb> Polynomial(const Indicator<lb, rb> & indicator)
        : origin(indicator.interval.center()),
          half_width(indicator.interval.width() / 2.),
          coefficients(make_span(indicator_polynomial_coefficients)) {}

    template <Bound lb, Bound rb> Polynomial(const Polynom<lb, rb> & polynom)
        : origin(polynom.interval.center()),
          half_width(polynom.interval.width() / 2.),
          coefficients(make_span(polynom.coefficients)) {}

    size_t degree() const { return coefficients.size() - 1; }
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
 * First the shiftings of p/q are extracted to outer_shifting (applied afterwards).
 * Then we consider p/q both centered on 0, which is a simpler case.
 * This generates 3 components, each polynomial on a "local" interval shifted from 0.
 * For each component, the convolution integral is expressed as a polynom of x-interval.center (shifted).
 * This is used to determine the coefficients of the component from the parameters of p/q.
 * Finally the component is created from its coefficients and "global" interval ("local" interval + outer_shift).
 *
 * 3 consecutive components defined on ]-hwp-hwq,-hwp+hwq] ]-hwp+hwq,hwp-hwq] ]hwp-hwq,hwqp+hwq] (outer_shift ignored).
 * Open-Closed bounds are chosen so the sum contains 1 value at internal border points.
 * Open-Closed is chosen instead of Closed-Open to better fit hawkes computation (cases with ]0,?]).
 * The Open bound on left component is ok, as left_part(-hwp-hwq) == 0 by construction of the formulas.
 */
inline void append_convolution_components(
    std::vector<Polynom<Bound::Open, Bound::Closed>> & components, Polynomial p, Polynomial q) {
    // Ensure q is the polynom with the smallest width
    if(p.half_width < q.half_width) {
        std::swap(p, q);
    }
    if(q.half_width == 0.) {
        return; // Optimization : convolution with zero width support is a zero value.
    }
    // Useful numerical tools / values
    const PointSpace outer_shifting = p.origin + q.origin;
    const size_t border_parts_degree = p.degree() + q.degree() + 1;
    const size_t center_part_degree = p.degree();
    const auto binomial = BinomialCoefficientsUpToN(border_parts_degree);
    const auto hwp_powers = PowersUpToN(border_parts_degree, p.half_width);
    const auto hwq_powers = PowersUpToN(border_parts_degree, q.half_width);
    const auto minus_1_power = [](size_t n) -> double { return n % 2 == 0 ? 1. : -1.; };
    // Left part on local interval ]-hwp-hwq, -hwp+hwq].
    {
        auto coefficients = std::vector<double>(border_parts_degree + 1, 0.);
        for(size_t k = 0; k <= p.degree(); ++k) {
            for(size_t j = 0; j <= q.degree(); ++j) {
                for(size_t i = 0; i <= k; ++i) {
                    for(size_t l = 0; l <= k - i; ++l) {
                        const double factor = p.coefficients[k] * q.coefficients[j] * hwp_powers(l) * binomial(i, k) *
                                              binomial(l, k - i) * minus_1_power(i + l) / double(i + j + 1);
                        coefficients[k + j + 1 - l] += factor;
                        coefficients[(k - i) - l] += factor * minus_1_power(i + j) * hwq_powers(i + j + 1);
                    }
                }
            }
        }
        components.emplace_back(
            (outer_shifting - p.half_width) + Interval<Bound::Open, Bound::Closed>{-q.half_width, q.half_width},
            std::move(coefficients));
    }
    // Center part on local interval ]-hwp+hwq,hwp-hwq].
    // Optimization : avoid computing it if zero width (would be discarded).
    if(p.half_width > q.half_width) {
        auto coefficients = std::vector<double>(center_part_degree + 1, 0.);
        for(size_t k = 0; k <= p.degree(); ++k) {
            for(size_t j = 0; j <= q.degree(); ++j) {
                for(size_t i = 0; i <= k; ++i) {
                    coefficients[k - i] += p.coefficients[k] * q.coefficients[j] * binomial(i, k) *
                                           hwq_powers(i + j + 1) * (1. - minus_1_power(i + j + 1)) * minus_1_power(i) /
                                           double(i + j + 1);
                }
            }
        }
        const PointSpace half_width = p.half_width - q.half_width;
        components.emplace_back(
            outer_shifting + Interval<Bound::Open, Bound::Closed>{-half_width, half_width}, std::move(coefficients));
    }
    // Right part on local interval ]hwp-hwq, hwp+hwq].
    {
        auto coefficients = std::vector<double>(border_parts_degree + 1, 0.);
        for(size_t k = 0; k <= p.degree(); ++k) {
            for(size_t j = 0; j <= q.degree(); ++j) {
                for(size_t i = 0; i <= k; ++i) {
                    for(size_t l = 0; l <= k - i; ++l) {
                        const double factor = p.coefficients[k] * q.coefficients[j] * hwp_powers(l) * binomial(i, k) *
                                              binomial(l, k - i) * minus_1_power(i) / double(i + j + 1);
                        coefficients[k + j + 1 - l] += -factor;
                        coefficients[(k - i) - l] += factor * hwq_powers(i + j + 1);
                    }
                }
            }
        }
        components.emplace_back(
            (outer_shifting + p.half_width) + Interval<Bound::Open, Bound::Closed>{-q.half_width, q.half_width},
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
    const Indicator<lb, rb> & lhs, Polynomial rhs) {
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
    const Indicator<lb, rb> & lhs, const Add<std::vector<Polynom<Bound::Open, Bound::Closed>>> & rhs) {
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
inline double integral(Polynomial p) {
    // int_R P = int_I sum_i a_i (x - I.center)^i
    //         = int_[-hw,hw] sum_i a_i x^i
    //         = sum_i a_i hw^(i+1) (1 - (-1)^i+1) / (i+1)
    //         = 2 * sum_i a_2i hw^(2i+1) / (2i+1)
    assert(p.coefficients.size() > 0);
    const double hw2 = p.half_width * p.half_width;
    // Horner strategy : sum_i a_2i hw^2i / (2i+1)
    size_t i = (p.coefficients.size() - 1) / 2;
    double r = p.coefficients[2 * i] / double(2 * i + 1);
    while(i > 0) {
        i -= 1;
        r = p.coefficients[2 * i] / double(2 * i + 1) + hw2 * r;
    }
    // Add final width multiplier
    return r * 2. * p.half_width;
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
