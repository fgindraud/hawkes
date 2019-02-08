#pragma once

#include <cmath>
#include <type_traits>

#include "types.h"

namespace shape {

using ::Point;
using ::PointSpace;

inline double square (double x) {
	return x * x;
}
inline double cube (double x) {
	return x * square (x);
}

// [left, right]
struct ClosedInterval {
	Point left;
	Point right;
	ClosedInterval () = default;
	ClosedInterval (Point left, Point right) : left (left), right (right) { assert (left <= right); }
};
inline ClosedInterval operator+ (PointSpace offset, const ClosedInterval i) {
	return {offset + i.left, offset + i.right};
}
inline ClosedInterval operator- (const ClosedInterval & i) {
	return {-i.right, -i.left};
}
inline bool operator== (const ClosedInterval & lhs, const ClosedInterval & rhs) {
	return lhs.left == rhs.left && lhs.right == rhs.right;
}
inline bool contains (const ClosedInterval & i, Point value) {
	return i.left <= value && value <= i.right;
}

/******************************************************************************
 * Combinators.
 */

// Invert the temporal space.
template <typename Inner> struct Reversed {
	Inner inner;

	ClosedInterval non_zero_domain () const { return -inner.non_zero_domain (); }
	double operator() (Point x) const { return inner (-x); }
};

// Temporal shift of a shape: move it forward by 'shift'.
template <typename Inner> struct Shifted {
	PointSpace shift;
	Inner inner;

	ClosedInterval non_zero_domain () const { return shift + inner.non_zero_domain (); }
	double operator() (Point x) const { return inner (x - shift); }
};

// Scale a shape on the vertical axis by 'scale'.
template <typename Inner> struct Scaled {
	double scale;
	Inner inner;

	ClosedInterval non_zero_domain () const { return inner.non_zero_domain (); }
	double operator() (Point x) const { return scale * inner (x); }
};

/* Priority value for combinator application.
 * This is used to force order of simplifications in convolution(a,b) or other modificators.
 * Without it, we have multiple valid overloads and ambiguity, leading to a compile error.
 * The higher the priority, the quickest a rule is applied.
 * FIXME simplify with set of overloads ?
 */
template <typename T> struct Priority { static constexpr int value = 0; };

template <typename Inner> struct Priority<Reversed<Inner>> { static constexpr int value = 1; };
template <typename Inner> struct Priority<Shifted<Inner>> { static constexpr int value = 2; };
template <typename Inner> struct Priority<Scaled<Inner>> { static constexpr int value = 3; };

template <typename Inner> inline auto reversed (const Inner & inner) {
	return Reversed<Inner>{inner};
}

template <typename Inner> inline auto shifted (PointSpace shift, const Inner & inner) {
	return Shifted<Inner>{shift, inner};
}
template <typename Inner> inline auto shifted (PointSpace shift, const Shifted<Inner> & s) {
	return shifted (shift + s.shift, s.inner);
}

template <typename Inner> inline auto scaled (double scale, const Inner & inner) {
	return Scaled<Inner>{scale, inner};
}
template <typename Inner> inline auto scaled (double scale, const Scaled<Inner> & s) {
	return scaled (scale * s.scale, s.inner);
}

// Component decomposition
template <typename Inner, typename ComponentTag> inline auto component (const Scaled<Inner> & shape, ComponentTag tag) {
	return scaled (shape.scale, component (shape.inner, tag));
}
template <typename Inner, typename ComponentTag>
inline auto component (const Shifted<Inner> & shape, ComponentTag tag) {
	return shifted (shape.shift, component (shape.inner, tag));
}

// Interval approximation
template <typename Inner> inline auto interval_approximation (const Scaled<Inner> & shape) {
	return scaled (shape.scale, interval_approximation (shape.inner));
}
template <typename Inner> inline auto interval_approximation (const Shifted<Inner> & shape) {
	return shifted (shape.shift, interval_approximation (shape.inner));
}

// Convolution simplifications: propagate combinators to the outer levels
template <typename L, typename R, typename = std::enable_if_t<(Priority<R>::value < 2)>>
inline auto convolution (const Shifted<L> & lhs, const R & rhs) {
	return shifted (lhs.shift, convolution (lhs.inner, rhs));
}
template <typename L, typename R, typename = std::enable_if_t<(Priority<L>::value <= 2)>>
inline auto convolution (const L & lhs, const Shifted<R> & rhs) {
	return shifted (rhs.shift, convolution (lhs, rhs.inner));
}

template <typename L, typename R, typename = std::enable_if_t<(Priority<R>::value < 3)>>
inline auto convolution (const Scaled<L> & lhs, const R & rhs) {
	return scaled (lhs.scale, convolution (lhs.inner, rhs));
}
template <typename L, typename R, typename = std::enable_if_t<(Priority<L>::value <= 3)>>
inline auto convolution (const L & lhs, const Scaled<R> & rhs) {
	return scaled (rhs.scale, convolution (lhs, rhs.inner));
}

/******************************************************************************
 * Base shapes.
 */

/* Indicator function for a closed interval.
 * Not normalized, returns 1. in the interval.
 */
struct IntervalIndicator {
	PointSpace half_width; // [0, inf[

	IntervalIndicator (PointSpace half_width) : half_width (half_width) { assert (half_width >= 0.); }
	static IntervalIndicator with_half_width (PointSpace half_width) { return {half_width}; }
	static IntervalIndicator with_width (PointSpace width) { return {width / 2.}; }

	ClosedInterval non_zero_domain () const { return {-half_width, half_width}; }
	double operator() (Point x) const {
		if (contains (non_zero_domain (), x)) {
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

	PositiveTriangle (PointSpace side) : side (side) { assert (side >= 0.); }

	ClosedInterval non_zero_domain () const { return {0., side}; }
	double operator() (Point x) const {
		if (contains (non_zero_domain (), x)) {
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

	NegativeTriangle (PointSpace side) : side (side) { assert (side >= 0.); }

	ClosedInterval non_zero_domain () const { return {-side, 0.}; }
	double operator() (Point x) const {
		if (contains (non_zero_domain (), x)) {
			return -x;
		} else {
			return 0.;
		}
	}
};
inline auto as_positive_triangle (NegativeTriangle t) {
	// NegativeTriangle(side)(x) == PositiveTriangle(side)(x)
	return reversed (PositiveTriangle{t.side});
}

/* Trapezoid with a block (2*half_base, height) with a PositiveTriangle(height) on the left and a negative on the right.
 */
struct Trapezoid {
	PointSpace height;    // [0, inf[
	PointSpace half_base; // [0, inf[
	PointSpace half_len;  // Precomputed

	Trapezoid (PointSpace height, PointSpace half_base) : height (height), half_base (half_base) {
		assert (height >= 0.);
		assert (half_base >= 0.);
		half_len = half_base + height;
	}

	ClosedInterval non_zero_domain () const { return {-half_len, half_len}; }
	double operator() (Point x) const {
		if (!contains (non_zero_domain (), x)) {
			return 0.;
		} else if (x < -half_base) {
			return x + half_len; // Left triangle
		} else if (x > half_base) {
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
inline auto component (const Trapezoid & trapezoid, Trapezoid::CentralBlock) {
	return scaled (trapezoid.height, IntervalIndicator::with_half_width (trapezoid.half_base));
}
inline auto component (const Trapezoid & trapezoid, Trapezoid::LeftTriangle) {
	return shifted (-trapezoid.half_len, PositiveTriangle{trapezoid.height});
}
inline auto component (const Trapezoid & trapezoid, Trapezoid::RightTriangle) {
	return shifted (trapezoid.half_len, NegativeTriangle{trapezoid.height});
}

inline Trapezoid convolution (const IntervalIndicator & left, const IntervalIndicator & right) {
	return Trapezoid{std::min (left.half_width, right.half_width) * 2, std::abs (left.half_width - right.half_width)};
}

inline auto interval_approximation (const Trapezoid & trapezoid) {
	return scaled (trapezoid.height, IntervalIndicator::with_half_width (trapezoid.half_len));
}

/* Convolution between IntervalIndicator(half_width=l/2) and PositiveTriangle(side=c).
 */
struct ConvolutionIntervalPositiveTriangle {
	PointSpace half_l; // [0, inf[
	PointSpace c;      // [0, inf[
	// Precomputed values
	PointSpace central_section_left;
	PointSpace central_section_right;

	ConvolutionIntervalPositiveTriangle (PointSpace half_l, PointSpace c) : half_l (half_l), c (c) {
		assert (half_l >= 0.);
		assert (c >= 0.);
		std::tie (central_section_left, central_section_right) = std::minmax (half_l, c - half_l);
		// Check bounds
		assert (-half_l <= central_section_left);
		assert (central_section_left <= central_section_right);
		assert (central_section_right <= c + half_l);
	}

	ClosedInterval non_zero_domain () const { return {-half_l, c + half_l}; }
	double operator() (Point x) const {
		if (!contains (non_zero_domain (), x)) {
			return 0.;
		} else if (x < central_section_left) {
			return square (x + half_l) / 2.; // Quadratic left part
		} else if (x > central_section_right) {
			return (square (c) - square (x - half_l)) / 2.; // Quadratic right part
		} else {
			// Central section has two behaviors depending on l <=> c
			if (2. * half_l >= c) {
				return square (c) / 2.; // l >= c : constant central part
			} else {
				return 2. * half_l * x; // l < c : linear central part
			}
		}
	}
};

inline auto convolution (const IntervalIndicator & lhs, const PositiveTriangle & rhs) {
	return ConvolutionIntervalPositiveTriangle (lhs.half_width, rhs.side);
}
inline auto convolution (const IntervalIndicator & lhs, const NegativeTriangle & rhs) {
	// IntervalIndicator is symmetric, and NegativeTriangle(x) == PositiveTriangle(-x)
	return reversed (convolution (lhs, PositiveTriangle{rhs.side}));
}
inline auto convolution (const PositiveTriangle & lhs, const IntervalIndicator & rhs) {
	return convolution (rhs, lhs);
}
inline auto convolution (const NegativeTriangle & lhs, const IntervalIndicator & rhs) {
	return convolution (rhs, lhs);
}

/* Convolution between PositiveTriangle(side=a) and PositiveTriangle(side=b).
 */
struct ConvolutionPositiveTrianglePositiveTriangle {
	// Precomputed values (definitions in the shape doc)
	PointSpace a_plus_b;
	PointSpace A;
	PointSpace B;
	double polynom_constant;

	ConvolutionPositiveTrianglePositiveTriangle (PointSpace a, PointSpace b) {
		assert (a >= 0.);
		assert (b >= 0.);
		a_plus_b = a + b;
		std::tie (A, B) = std::minmax (a, b);
		// Polynom constant used for cubic right part = -2a^2 -2b^2 + 2ab.
		polynom_constant = 2. * (3. * a * b - square (a_plus_b));
		// Check bounds
		assert (0. <= A);
		assert (A <= B);
		assert (B <= a_plus_b);
	}

	ClosedInterval non_zero_domain () const { return {0., a_plus_b}; }
	double operator() (Point x) const {
		if (!contains (non_zero_domain (), x)) {
			return 0.;
		} else if (x < A) {
			return cube (x) / 6.; // Cubic left part
		} else if (x > B) {
			// Cubic right part
			const auto polynom = x * (x + a_plus_b) + polynom_constant;
			return (a_plus_b - x) * polynom / 6.;
		} else {
			return square (A) * (3. * x - 2. * A) / 6.; // Central section has one formula using A=min(a,b)
		}
	}
};

inline auto convolution (const PositiveTriangle & lhs, const PositiveTriangle & rhs) {
	return ConvolutionPositiveTrianglePositiveTriangle (lhs.side, rhs.side);
}
inline auto convolution (const NegativeTriangle & lhs, const NegativeTriangle & rhs) {
	// Reversing the time dimension transforms both negative triangles into positive ones.
	return reversed (convolution (PositiveTriangle{lhs.side}, PositiveTriangle{rhs.side}));
}

/* Convolution between NegativeTriangle(side=a) and PositiveTriangle(side=b).
 */
struct ConvolutionNegativeTrianglePositiveTriangle {
	PointSpace a;
	PointSpace b;
	// Precomputed values
	PointSpace A;
	PointSpace B;

	ConvolutionNegativeTrianglePositiveTriangle (PointSpace a, PointSpace b) : a (a), b (b) {
		assert (a >= 0.);
		assert (b >= 0.);
		std::tie (A, B) = std::minmax (0., b - a);
		// Check bounds
		assert (-a <= A);
		assert (A <= B);
		assert (B <= b);
	}

	ClosedInterval non_zero_domain () const { return {-a, b}; }
	double operator() (Point x) const {
		if (!contains (non_zero_domain (), x)) {
			return 0.;
		} else if (x < A) {
			return square (x + a) * (2. * a - x) / 6.; // Cubic left part
		} else if (x > B) {
			return square (b - x) * (2. * b + x) / 6.; // Cubic right part
		} else {
			// Central section has two behaviors depending on a <=> b
			if (a < b) {
				return square (a) * (2. * a + 3. * x) / 6.; // Linear central part for [0, b-a].
			} else {
				return square (b) * (2. * b - 3. * x) / 6.; // Linear central part for [b-a, 0].
			}
		}
	}
};

inline auto convolution (const NegativeTriangle & lhs, const PositiveTriangle & rhs) {
	return ConvolutionNegativeTrianglePositiveTriangle (lhs.side, rhs.side);
}
inline auto convolution (const PositiveTriangle & lhs, const NegativeTriangle & rhs) {
	return convolution (rhs, lhs);
}
} // namespace shape
