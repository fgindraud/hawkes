#pragma once

#include <cmath>
#include <tuple>       // Add
#include <type_traits> // enable_if
#include <utility>

#include <limits>

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

inline double square(double x) {
    return x * x;
}
inline double cube(double x) {
    return x * square(x);
}

// TODO remove
using ClosedInterval = Interval<Bound::Closed, Bound::Closed>;

// tuple-only equivalent of std::apply, used in add.
template <typename F, size_t... Is, typename... Types>
inline auto tuple_apply_impl(F && f, const std::tuple<Types...> & t, std::index_sequence<Is...>) {
    return std::forward<F>(f)(std::get<Is>(t)...);
}
template <typename F, typename... Types> inline auto tuple_apply(F && f, const std::tuple<Types...> & t) {
    return tuple_apply_impl(std::forward<F>(f), t, std::index_sequence_for<Types...>());
}

/******************************************************************************
 * Combinators.
 */

// Invert the temporal space.
template <typename Inner> struct Reversed {
    Inner inner;

    ClosedInterval non_zero_domain() const { return -inner.non_zero_domain(); }
    double operator()(Point x) const { return inner(-x); }
};
template <typename Inner> inline auto reversed(const Inner & inner) {
    return Reversed<Inner>{inner};
}

// Temporal shift of a shape: move it forward by 'shift'.
template <typename Inner> struct Shifted {
    PointSpace shift;
    Inner inner;

    ClosedInterval non_zero_domain() const { return shift + inner.non_zero_domain(); }
    double operator()(Point x) const { return inner(x - shift); }
};
template <typename Inner> inline auto shifted(PointSpace shift, const Inner & inner) {
    return Shifted<Inner>{shift, inner};
}
template <typename Inner> inline auto shifted(PointSpace shift, const Shifted<Inner> & s) {
    return shifted(shift + s.shift, s.inner);
}

// Scale a shape on the vertical axis by 'scale'.
template <typename Inner> struct Scaled {
    double scale;
    Inner inner;

    ClosedInterval non_zero_domain() const { return inner.non_zero_domain(); }
    double operator()(Point x) const { return scale * inner(x); }
};
template <typename Inner> inline auto scaled(double scale, const Inner & inner) {
    return Scaled<Inner>{scale, inner};
}
template <typename Inner> inline auto scaled(double scale, const Scaled<Inner> & s) {
    return scaled(scale * s.scale, s.inner);
}

// Addition of multiple sub-shapes
inline double add_doubles(double x) {
    return x;
}
template <typename... Doubles> inline double add_doubles(double first, Doubles... others) {
    return first + add_doubles(others...);
}
template <typename... Shapes> struct Add {
    std::tuple<Shapes...> shapes;
    ClosedInterval non_zero_domain_; // Precomputed

    Add(const Shapes &... from_shapes) : shapes(from_shapes...) {
        non_zero_domain_.left = std::min({from_shapes.non_zero_domain().left...});
        non_zero_domain_.right = std::max({from_shapes.non_zero_domain().right...});
        assert(non_zero_domain_.left <= non_zero_domain_.right);
    }

    ClosedInterval non_zero_domain() const { return non_zero_domain_; }
    double operator()(Point x) const {
        if(!non_zero_domain_.contains(x)) {
            return 0.;
        } else {
            return tuple_apply([x](const Shapes &... shapes) { return add_doubles(shapes(x)...); }, shapes);
        }
    }
};
template <typename... Shapes> inline auto add(const Shapes &... shapes) {
    return Add<Shapes...>(shapes...);
}

/* Simplification of expressions.
 *
 * convolution(a,b) is defined for each specific shape pair.
 * Shapeis modified by combinators are handled using the convolution rules, like:
 * convolution(shifted(a), b) = shifted(convolution(a,b))
 * convolution(a, scaled(b)) = scaled(convolution(a,b))
 * etc,...
 *
 * Simplifications for other operations are defined in the same way, like for component().
 */

/******************************************************************************
 * Base shapes.
 */

/* Indicator function for a closed interval.
 * Not normalized, returns 1. in the interval.
 */
struct IntervalIndicator {
    PointSpace half_width; // [0, inf[

    IntervalIndicator(PointSpace half_width) : half_width(half_width) { assert(half_width >= 0.); }
    static IntervalIndicator with_half_width(PointSpace half_width) { return {half_width}; }
    static IntervalIndicator with_width(PointSpace width) { return {width / 2.}; }

    ClosedInterval non_zero_domain() const { return {-half_width, half_width}; }
    double operator()(Point x) const {
        if(non_zero_domain().contains(x)) {
            return 1.;
        } else {
            return 0.;
        }
    }
};

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

    // Component type tags.
    struct LeftTriangle {};
    struct CentralBlock {};
    struct RightTriangle {};
};

// Decompose into components
inline auto component(const Trapezoid & trapezoid, Trapezoid::CentralBlock) {
    return scaled(
        trapezoid.height, Indicator<Bound::Closed, Bound::Closed>{{-trapezoid.half_base, trapezoid.half_base}});
}
inline auto component(const Trapezoid & trapezoid, Trapezoid::LeftTriangle) {
    return shifted(-trapezoid.half_len, PositiveTriangle{trapezoid.height});
}
inline auto component(const Trapezoid & trapezoid, Trapezoid::RightTriangle) {
    return shifted(trapezoid.half_len, NegativeTriangle{trapezoid.height});
}

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

/******************************************************************************
 * NEW SHAPE impl
 */

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

/******************************************************************************
 * Shape manipulation functions and combinators.
 * Combinators : modify a shape, like x/y translation, x/y scaling, ...
 *
 * Strategy :
 * Define explicit cases for base shapes.
 * Define simplifications when combinators are found.
 *
 * The conventional order of combinators is:
 * scaling -> shifting -> reversion -> base_shape
 */

/* Combinator: Reverse x dimension
 * reverse(f)(x) = f(-x)
 */
template <Bound lb, Bound rb> Indicator<rb, lb> reverse(const Indicator<lb, rb> & indicator) {
    return {-indicator.interval};
}

/* Combinator: Shift x dimension
 * shifted(s, f)(x) = f(x - s)
 */
template <Bound lb, Bound rb> Indicator<rb, lb> shifted(PointSpace s, const Indicator<lb, rb> & indicator) {
    return {s + indicator.interval};
}

/* Convolution.
 * convolution(f,g)(x) = int_R f(x - t) g (t) dt
 *
 * Perform simplification by factoring scaling and shifting before applying base shape convolution.
 */
template <typename Lhs, typename Rhs> inline auto convolution(const Lhs & lhs, const Rhs & rhs) {
    return convolution_extract_scale(lhs, rhs);
}

// Extract scale
template <typename L, typename R> inline auto convolution_extract_scale(const Scaled<L> & lhs, const Scaled<R> & rhs) {
    return scaled(lhs.scale * rhs.scale, convolution_extract_shift(lhs.inner, rhs.inner));
}
template <typename L, typename R> inline auto convolution_extract_scale(const Scaled<L> & lhs, const R & rhs) {
    return scaled(lhs.scale, convolution_extract_shift(lhs.inner, rhs));
}
template <typename L, typename R> inline auto convolution_extract_scale(const L & lhs, const Scaled<R> & rhs) {
    return scaled(rhs.scale, convolution_extract_shift(lhs, rhs.inner));
}
template <typename L, typename R> inline auto convolution_extract_scale(const L & lhs, const R & rhs) {
    return convolution_extract_shift(lhs, rhs);
}

// Extract shift
template <typename L, typename R>
inline auto convolution_extract_shift(const Shifted<L> & lhs, const Shifted<R> & rhs) {
    return shifted(lhs.shift + rhs.shift, convolution_base(lhs.inner, rhs.inner));
}
template <typename L, typename R> inline auto convolution_extract_shift(const Shifted<L> & lhs, const R & rhs) {
    return shifted(lhs.shift, convolution_base(lhs.inner, rhs));
}
template <typename L, typename R> inline auto convolution_extract_shift(const L & lhs, const Shifted<R> & rhs) {
    return shifted(rhs.shift, convolution_base(lhs, rhs.inner));
}
template <typename L, typename R> inline auto convolution_extract_shift(const L & lhs, const R & rhs) {
    return convolution_base(lhs, rhs);
}

// Base cases
inline Trapezoid convolution_base(const IntervalIndicator & left, const IntervalIndicator & right) {
    return Trapezoid{std::min(left.half_width, right.half_width) * 2, std::abs(left.half_width - right.half_width)};
}

inline auto convolution_base(const IntervalIndicator & lhs, const PositiveTriangle & rhs) {
    return ConvolutionIntervalPositiveTriangle(lhs.half_width, rhs.side);
}
inline auto convolution_base(const IntervalIndicator & lhs, const NegativeTriangle & rhs) {
    // IntervalIndicator is symmetric, and NegativeTriangle(x) == PositiveTriangle(-x)
    return reversed(convolution_base(lhs, PositiveTriangle{rhs.side}));
}
inline auto convolution_base(const PositiveTriangle & lhs, const IntervalIndicator & rhs) {
    return convolution_base(rhs, lhs);
}
inline auto convolution_base(const NegativeTriangle & lhs, const IntervalIndicator & rhs) {
    return convolution_base(rhs, lhs);
}

inline auto convolution_base(const PositiveTriangle & lhs, const PositiveTriangle & rhs) {
    return ConvolutionPositiveTrianglePositiveTriangle(lhs.side, rhs.side);
}
inline auto convolution_base(const NegativeTriangle & lhs, const NegativeTriangle & rhs) {
    // Reversing the time dimension transforms both negative triangles into positive ones.
    return reversed(convolution_base(PositiveTriangle{lhs.side}, PositiveTriangle{rhs.side}));
}

inline auto convolution_base(const NegativeTriangle & lhs, const PositiveTriangle & rhs) {
    return ConvolutionNegativeTrianglePositiveTriangle(lhs.side, rhs.side);
}
inline auto convolution_base(const PositiveTriangle & lhs, const NegativeTriangle & rhs) {
    return convolution_base(rhs, lhs);
}

inline auto convolution_base(const Trapezoid & lhs, const IntervalIndicator & rhs) {
    return add(
        convolution(component(lhs, Trapezoid::LeftTriangle{}), rhs),
        convolution(component(lhs, Trapezoid::CentralBlock{}), rhs),
        convolution(component(lhs, Trapezoid::RightTriangle{}), rhs));
}
inline auto convolution_base(const IntervalIndicator & lhs, const Trapezoid & rhs) {
    return convolution_base(rhs, lhs);
}
inline auto convolution_base(const Trapezoid & lhs, const Trapezoid & rhs) {
    const auto left_part = Trapezoid::LeftTriangle{};
    const auto central_part = Trapezoid::CentralBlock{};
    const auto right_part = Trapezoid::RightTriangle{};
    return add(
        //
        convolution(component(lhs, left_part), component(rhs, left_part)),
        convolution(component(lhs, central_part), component(rhs, left_part)),
        convolution(component(lhs, right_part), component(rhs, left_part)),
        //
        convolution(component(lhs, left_part), component(rhs, central_part)),
        convolution(component(lhs, central_part), component(rhs, central_part)),
        convolution(component(lhs, right_part), component(rhs, central_part)),
        //
        convolution(component(lhs, left_part), component(rhs, right_part)),
        convolution(component(lhs, central_part), component(rhs, right_part)),
        convolution(component(lhs, right_part), component(rhs, right_part)));
}

/* Cross correlation.
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
    return scaled(lhs.scale * rhs.scale, cross_correlation_extract_shift(lhs.inner, rhs.inner));
}
template <typename L, typename R> inline auto cross_correlation_extract_scale(const Scaled<L> & lhs, const R & rhs) {
    return scaled(lhs.scale, cross_correlation_extract_shift(lhs.inner, rhs));
}
template <typename L, typename R> inline auto cross_correlation_extract_scale(const L & lhs, const Scaled<R> & rhs) {
    return scaled(rhs.scale, cross_correlation_extract_shift(lhs, rhs.inner));
}
template <typename L, typename R> inline auto cross_correlation_extract_scale(const L & lhs, const R & rhs) {
    return cross_correlation_extract_shift(lhs, rhs);
}

// Extract shift
template <typename L, typename R>
inline auto cross_correlation_extract_shift(const Shifted<L> & lhs, const Shifted<R> & rhs) {
    return shifted(rhs.shift - lhs.shift, cross_correlation_base(lhs.inner, rhs.inner));
}
template <typename L, typename R> inline auto cross_correlation_extract_shift(const Shifted<L> & lhs, const R & rhs) {
    return shifted(-lhs.shift, cross_correlation_base(lhs.inner, rhs));
}
template <typename L, typename R> inline auto cross_correlation_extract_shift(const L & lhs, const Shifted<R> & rhs) {
    return shifted(rhs.shift, cross_correlation_base(lhs, rhs.inner));
}
template <typename L, typename R> inline auto cross_correlation_extract_shift(const L & lhs, const R & rhs) {
    return cross_correlation_base(lhs, rhs);
}

// Base cases
inline Trapezoid cross_correlation_base(const IntervalIndicator & left, const IntervalIndicator & right) {
    return convolution_base(left, right); // Indicator is symmetric, identical by time inversion.
}
inline auto cross_correlation_base(const Trapezoid & lhs, const Trapezoid & rhs) {
    return convolution_base(lhs, rhs); // Trapezoid is symmetric, identical by time inversion
}

/* Approximate a shape with an interval.
 */
template <typename Inner> inline auto interval_approximation(const Scaled<Inner> & shape) {
    return scaled(shape.scale, interval_approximation(shape.inner));
}
template <typename Inner> inline auto interval_approximation(const Shifted<Inner> & shape) {
    return shifted(shape.shift, interval_approximation(shape.inner));
}
inline auto interval_approximation(const Trapezoid & trapezoid) {
    return scaled(trapezoid.height, IntervalIndicator::with_half_width(trapezoid.half_len));
}

/* Component decomposition.
 */
template <typename Inner, typename ComponentTag> inline auto component(const Scaled<Inner> & shape, ComponentTag tag) {
    return scaled(shape.scale, component(shape.inner, tag));
}
template <typename Inner, typename ComponentTag> inline auto component(const Shifted<Inner> & shape, ComponentTag tag) {
    return shifted(shape.shift, component(shape.inner, tag));
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

// Legacy wrapper TODO remove
inline double sup_of_sum_of_differences_to_points(
    const SortedVec<Point> & points, const IntervalIndicator & indicator) {
    return sup_of_sum_of_differences_to_points(
        points, Indicator<Bound::Closed, Bound::Closed>{indicator.non_zero_domain()});
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
// Shifting has no effect on the sup value.
template <typename Inner>
inline double sup_of_sum_of_differences_to_points(const SortedVec<Point> & points, const Shifted<Inner> & shape) {
    return sup_of_sum_of_differences_to_points(points, shape.inner);
}

} // namespace shape
