#pragma once

#include <cmath>
#include <type_traits>

#include "types.h"

namespace shape {

using ::Point;
using std::int32_t;
using std::int64_t;
static_assert (std::is_same<Point, int32_t>::value, "Point must be an int32_t to avoid any overflow");

// [from, to]
template <typename T> struct ClosedInterval {
	T from;
	T to;
};
template <typename T> inline ClosedInterval<T> operator+ (const T & offset, const ClosedInterval<T> & i) {
	return {offset + i.from, offset + i.to};
}
template <typename T> inline ClosedInterval<T> operator- (const ClosedInterval<T> & i) {
	return {-i.to, -i.from};
}
template <typename T> inline bool contains (const ClosedInterval<T> & i, const T & value) {
	return i.from <= value && value <= i.to;
}

// Type tag to indicate that a value is supposed to be in the non zero domain.
struct PointInNonZeroDomain {
	Point value;
};

// Priority flag
template <typename T> struct Priority { static constexpr int value = 0; };

// Temporal shift of a shape: move it forward by 'shift'.
template <typename Inner> struct Shifted {
	int32_t shift;
	Inner inner;

	ClosedInterval<Point> non_zero_domain () const { return shift + inner.non_zero_domain (); }
	auto operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		return inner (PointInNonZeroDomain{x.value - shift});
	}
	auto operator() (Point x) const { return inner (x - shift); }
};
template <typename Inner> struct Priority<Shifted<Inner>> { static constexpr int value = 1; };
template <typename Inner> inline auto shifted (int32_t shift, Inner inner) {
	return Shifted<Inner>{shift, inner};
}

// Convolution simplifications: propagate shift to the outer levels
template <typename L, typename R, typename = std::enable_if_t<(Priority<R>::value < 1)>>
inline auto convolution (const Shifted<L> & lhs, const R & rhs) {
	return shifted (lhs.shift, convolution (lhs.inner, rhs));
}
template <typename L, typename R, typename = std::enable_if_t<(Priority<L>::value <= 1)>>
inline auto convolution (const L & lhs, const Shifted<R> & rhs) {
	return shifted (rhs.shift, convolution (lhs, rhs.inner));
}

// Scale a shape on the vertical axis by 'scale'.
template <typename Inner> struct Scaled {
	int64_t scale;
	Inner inner;

	ClosedInterval<Point> non_zero_domain () const { return inner.non_zero_domain (); }
	auto operator() (PointInNonZeroDomain x) const { return scale * inner (x); }
	auto operator() (Point x) const { return scale * inner (x); }
};
template <typename Inner> struct Priority<Scaled<Inner>> { static constexpr int value = 2; };
template <typename Inner> inline auto scaled (int64_t scale, Inner inner) {
	return Scaled<Inner>{scale, inner};
}

// Convolution simplification: propagate scaling to the outer levels
template <typename L, typename R, typename = std::enable_if_t<(Priority<R>::value < 2)>>
inline auto convolution (const Scaled<L> & lhs, const R & rhs) {
	return scaled (lhs.scale, convolution (lhs.inner, rhs));
}
template <typename L, typename R, typename = std::enable_if_t<(Priority<L>::value <= 2)>>
inline auto convolution (const L & lhs, const Scaled<R> & rhs) {
	return scaled (rhs.scale, convolution (lhs, rhs.inner));
}

// Indicator function for an interval.
struct IntervalIndicator {
	int32_t half_width; // [0, inf[

	static IntervalIndicator with_half_width (int32_t half_width) { return {half_width}; }
	static IntervalIndicator with_width (int32_t width) { return {width / 2}; }

	ClosedInterval<Point> non_zero_domain () const { return {-half_width, half_width}; }
	int32_t operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		return 1;
	}
	int32_t operator() (Point x) const {
		return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0;
	}
};

// Triangle (0,0), (side, 0), (side, side)
struct PositiveTriangle {
	int32_t side; // [0, inf[

	ClosedInterval<Point> non_zero_domain () const { return {0, side}; }
	int32_t operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		return x.value;
	}
	int32_t operator() (Point x) const {
		return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0;
	}
};

// Triangle (0,0), (-side, 0), (-side, side)
struct NegativeTriangle {
	int32_t side; // [0, inf[

	ClosedInterval<Point> non_zero_domain () const { return {-side, 0}; }
	int32_t operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		return -x.value;
	}
	int32_t operator() (Point x) const {
		return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0;
	}
};

// Trapezoid with a block (2*half_base, height) with a PositiveTriangle(height) on the left and a negative on the right.
struct Trapezoid {
	int32_t height;    // [0, inf[
	int32_t half_base; // [0, inf[

	ClosedInterval<Point> non_zero_domain () const {
		const auto half_len = height + half_base;
		return {-half_len, half_len};
	}
	int32_t operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		if (x.value < -half_base) {
			return x.value - (half_base + height); // Left triangle
		} else if (x.value > half_base) {
			return (half_base + height) - x.value; // Right triangle
		} else {
			return height; // Central block
		}
	}
	int32_t operator() (Point x) const {
		return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0;
	}
};

inline auto central_block (const Trapezoid & trapezoid) {
	return scaled (trapezoid.height, IntervalIndicator::with_half_width (trapezoid.half_base));
}
inline auto left_triangle (const Trapezoid & trapezoid) {
	return shifted (-(trapezoid.height + trapezoid.half_base), PositiveTriangle{trapezoid.height});
}
inline auto right_triangle (const Trapezoid & trapezoid) {
	return shifted (trapezoid.height + trapezoid.half_base, NegativeTriangle{trapezoid.height});
}

inline Trapezoid convolution (const IntervalIndicator & left, const IntervalIndicator & right) {
	return {std::min (left.half_width, right.half_width) * 2, std::abs (left.half_width - right.half_width)};
}

// Convolution between IntervalIndicator(half_width=l/2) and PositiveTriangle(side=c)
struct ConvolutionIntervalPositiveTriangle {
	int32_t half_l;
	int32_t c;
	ClosedInterval<int32_t> central_section; // Cached boundaries for the central section

	ConvolutionIntervalPositiveTriangle (int32_t half_l, int32_t c) : half_l (half_l), c (c) {
		const auto p = std::minmax (half_l, c - half_l);
		central_section = {p.first, p.second};
	}

	ClosedInterval<Point> non_zero_domain () const { return {-half_l, c + half_l}; }
	double operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		if (x.value < central_section.from) {
			// Quadratic left part
			const int64_t sx = x.value + half_l;
			return double(sx * sx) * 0.5;
		} else if (x.value > central_section.to) {
			// Quadratic right part
			const int64_t sx = x.value - half_l;
			return double(int64_t (c) * int64_t (c) - sx * sx) / 2.;
		} else {
			// Central section has two behaviors depending on l <=> c
			if (2 * half_l >= c) {
				return double(int64_t (c) * int64_t (c)) / 2.; // l >= c : constant central part
			} else {
				return double(int64_t (x.value) * int64_t (2 * half_l)); // l < c : linear central part
			}
		}
	}
	double operator() (Point x) const {
		return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0.;
	}
};

inline ConvolutionIntervalPositiveTriangle convolution (const IntervalIndicator & lhs, const PositiveTriangle & rhs) {
	return {lhs.half_width, rhs.side};
}
inline ConvolutionIntervalPositiveTriangle convolution (const PositiveTriangle & lhs, const IntervalIndicator & rhs) {
	return convolution (rhs, lhs);
}

} // namespace shape
