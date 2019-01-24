#pragma once

#include <cmath>
#include <type_traits>

#include "types.h"

namespace shape {

using ::Point;
using ::PointSpace;
using std::int32_t;
using std::int64_t;
static_assert (std::is_same<Point, int32_t>::value, "Point must be an int32_t to avoid any overflow");
static_assert (std::is_same<PointSpace, int32_t>::value, "PointSpace must be int32_t for int size coherence");

inline int64_t square (int32_t x) {
	return int64_t (x) * int64_t (x); // Cannot overflow
}

// [left, right]
template <typename T> struct ClosedInterval {
	T left;
	T right;
	ClosedInterval () = default;
	ClosedInterval (T left, T right) : left (left), right (right) { assert (left <= right); }
};
template <typename T> inline ClosedInterval<T> operator+ (const T & offset, const ClosedInterval<T> & i) {
	return {offset + i.left, offset + i.right};
}
template <typename T> inline ClosedInterval<T> operator- (const ClosedInterval<T> & i) {
	return {-i.right, -i.left};
}
template <typename T> inline bool operator== (const ClosedInterval<T> & lhs, const ClosedInterval<T> & rhs) {
	return lhs.left == rhs.left && lhs.right == rhs.right;
}
template <typename T> inline bool contains (const ClosedInterval<T> & i, const T & value) {
	return i.left <= value && value <= i.right;
}

// Type tag to indicate that a value is supposed to be in the non zero domain.
struct PointInNonZeroDomain {
	Point value;
};

/******************************************************************************
 * Combinators.
 */

// Invert the temporal space.
template <typename Inner> struct Reversed {
	Inner inner;

	ClosedInterval<Point> non_zero_domain () const { return -inner.non_zero_domain (); }
	auto operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		return inner (PointInNonZeroDomain{-x.value});
	}
	auto operator() (Point x) const { return inner (-x); }
};

// Temporal shift of a shape: move it forward by 'shift'.
template <typename Inner> struct Shifted {
	PointSpace shift;
	Inner inner;

	ClosedInterval<Point> non_zero_domain () const { return shift + inner.non_zero_domain (); }
	auto operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		return inner (PointInNonZeroDomain{x.value - shift});
	}
	auto operator() (Point x) const { return inner (x - shift); }
};

// Scale a shape on the vertical axis by 'scale'.
template <typename T, typename Inner> struct Scaled {
	T scale;
	Inner inner;

	ClosedInterval<Point> non_zero_domain () const { return inner.non_zero_domain (); }
	auto operator() (PointInNonZeroDomain x) const { return scale * inner (x); }
	auto operator() (Point x) const { return scale * inner (x); }
};

/* Priority value for combinator application.
 * This is used to force order of simplifications in convolution(a,b) or other modificators.
 * Without it, we have multiple valid overloads and ambiguity, leading to a compile error.
 * The higher the priority, the quickest a rule is applied.
 */
template <typename T> struct Priority { static constexpr int value = 0; };

template <typename Inner> struct Priority<Reversed<Inner>> { static constexpr int value = 1; };
template <typename Inner> struct Priority<Shifted<Inner>> { static constexpr int value = 2; };
template <typename T, typename Inner> struct Priority<Scaled<T, Inner>> { static constexpr int value = 3; };

template <typename Inner> inline auto reversed (const Inner & inner) {
	return Reversed<Inner>{inner};
}

template <typename Inner> inline auto shifted (PointSpace shift, const Inner & inner) {
	return Shifted<Inner>{shift, inner};
}
template <typename Inner> inline auto shifted (PointSpace shift, const Shifted<Inner> & s) {
	return shifted (shift + s.shift, s.inner);
}

template <typename T, typename Inner> inline auto scaled (T scale, const Inner & inner) {
	return Scaled<T, Inner>{scale, inner};
}
template <typename T, typename U, typename Inner> inline auto scaled (T scale, const Scaled<U, Inner> & s) {
	return scaled (scale * s.scale, s.inner);
}

// Component decomposition
template <typename T, typename Inner, typename ComponentTag>
inline auto component (const Scaled<T, Inner> & shape, ComponentTag tag) {
	return scaled (shape.scale, component (shape.inner, tag));
}
template <typename Inner, typename ComponentTag>
inline auto component (const Shifted<Inner> & shape, ComponentTag tag) {
	return shifted (shape.shift, component (shape.inner, tag));
}

// Interval approximation
template <typename T, typename Inner> inline auto interval_approximation (const Scaled<T, Inner> & shape) {
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

template <typename T, typename L, typename R, typename = std::enable_if_t<(Priority<R>::value < 3)>>
inline auto convolution (const Scaled<T, L> & lhs, const R & rhs) {
	return scaled (lhs.scale, convolution (lhs.inner, rhs));
}
template <typename L, typename T, typename R, typename = std::enable_if_t<(Priority<L>::value <= 3)>>
inline auto convolution (const L & lhs, const Scaled<T, R> & rhs) {
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

	IntervalIndicator (PointSpace half_width) : half_width (half_width) { assert (half_width >= 0); }
	static IntervalIndicator with_half_width (PointSpace half_width) { return {half_width}; }
	static IntervalIndicator with_width (PointSpace width) { return {width / 2}; }

	ClosedInterval<Point> non_zero_domain () const { return {-half_width, half_width}; }
	int32_t operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		static_cast<void> (x); // Ignored, only used in assert
		return 1;
	}
	int32_t operator() (Point x) const {
		return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0;
	}
};

/* Triangle (0,0), (side, 0), (side, side).
 */
struct PositiveTriangle {
	PointSpace side; // [0, inf[

	PositiveTriangle (PointSpace side) : side (side) { assert (side >= 0); }

	ClosedInterval<Point> non_zero_domain () const { return {0, side}; }
	int32_t operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		return x.value;
	}
	int32_t operator() (Point x) const {
		return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0;
	}
};

/* Triangle (0,0), (-side, 0), (-side, side).
 */
struct NegativeTriangle {
	PointSpace side; // [0, inf[

	NegativeTriangle (PointSpace side) : side (side) { assert (side >= 0); }

	ClosedInterval<Point> non_zero_domain () const { return {-side, 0}; }
	int32_t operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		return -x.value;
	}
	int32_t operator() (Point x) const {
		return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0;
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
		assert (height >= 0);
		assert (half_base >= 0);
		half_len = half_base + height;
	}

	ClosedInterval<Point> non_zero_domain () const { return {-half_len, half_len}; }
	int32_t operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		if (x.value < -half_base) {
			return x.value + half_len; // Left triangle
		} else if (x.value > half_base) {
			return half_len - x.value; // Right triangle
		} else {
			return height; // Central block
		}
	}
	int32_t operator() (Point x) const {
		return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0;
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
	return {std::min (left.half_width, right.half_width) * 2, std::abs (left.half_width - right.half_width)};
}

inline auto interval_approximation (const Trapezoid & trapezoid) {
	return scaled (trapezoid.height, IntervalIndicator::with_half_width (trapezoid.half_len));
}

/* Convolution between IntervalIndicator(half_width=l/2) and PositiveTriangle(side=c).
 */
struct ConvolutionIntervalPositiveTriangle {
	PointSpace half_l;                          // [0, inf[
	PointSpace c;                               // [0, inf[
	ClosedInterval<PointSpace> central_section; // Precomputed values

	ConvolutionIntervalPositiveTriangle (PointSpace half_l, PointSpace c) : half_l (half_l), c (c) {
		assert (half_l >= 0);
		assert (c >= 0);
		const auto p = std::minmax (half_l, c - half_l);
		central_section = {p.first, p.second};
	}

	ClosedInterval<Point> non_zero_domain () const { return {-half_l, c + half_l}; }
	double operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		if (x.value < central_section.left) {
			// Quadratic left part
			return double(square (x.value + half_l)) * 0.5;
		} else if (x.value > central_section.right) {
			// Quadratic right part
			return double(square (c) - square (x.value - half_l)) / 2.;
		} else {
			// Central section has two behaviors depending on l <=> c
			if (2 * half_l >= c) {
				return double(square (c)) / 2.; // l >= c : constant central part
			} else {
				return double(int64_t (x.value) * int64_t (2 * half_l)); // l < c : linear central part
			}
		}
	}
	double operator() (Point x) const {
		return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0.;
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
	int64_t polynom_constant;

	ConvolutionPositiveTrianglePositiveTriangle (PointSpace a, PointSpace b) {
		assert (a >= 0);
		assert (b >= 0);
		a_plus_b = a + b;
		std::tie (A, B) = std::minmax (a, b);
		// Polynom constant used for cubic right part = -2a^2 -2b^2 + 2ab.
		polynom_constant = 2 * (3 * int64_t (a) * int64_t (b) - square (a_plus_b));
	}

	ClosedInterval<Point> non_zero_domain () const { return {0, a_plus_b}; }
	double operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		if (x.value < A) {
			// Cubic left part
			return double(square (x.value)) * double(x.value) / 6.;
		} else if (x.value > B) {
			// Cubic right part
			const int64_t polynom = int64_t (x.value) * int64_t (x.value + a_plus_b) + polynom_constant;
			return double(a_plus_b - x.value) * double(polynom) / 6.;
		} else {
			// Central section has one formula using A=min(a,b)
			return double(square (A)) * double(3 * x.value - 2 * A) / 6.;
		}
	}
	double operator() (Point x) const {
		return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0.;
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
		assert (a >= 0);
		assert (b >= 0);
		std::tie (A, B) = std::minmax (0, b - a);
	}

	ClosedInterval<Point> non_zero_domain () const { return {-a, b}; }
	double operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		if (x.value < A) {
			// Cubic left part
			return double(square (x.value + a)) * double(2 * a - x.value) / 6.;
		} else if (x.value > B) {
			// Cubic right part
			return double(square (b - x.value)) * double(2 * b + x.value) / 6.;
		} else {
			// Central section has two behaviors depending on a <=> b
			if (a < b) {
				// Linear central part for [0, b-a].
				return double(square (a)) * double(2 * a + 3 * x.value) / 6.;
			} else {
				// Linear central part for [b-a, 0].
				return double(square (b)) * double(2 * b - 3 * x.value) / 6.;
			}
		}
	}
	double operator() (Point x) const {
		return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0.;
	}
};

inline auto convolution (const NegativeTriangle & lhs, const PositiveTriangle & rhs) {
	return ConvolutionNegativeTrianglePositiveTriangle (lhs.side, rhs.side);
}
inline auto convolution (const PositiveTriangle & lhs, const NegativeTriangle & rhs) {
	return convolution (rhs, lhs);
}
} // namespace shape
