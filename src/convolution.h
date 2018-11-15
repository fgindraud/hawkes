#pragma once

#include <cmath>

#include "types.h"

namespace shape {

using ::Point;

// [from, to]
template <typename T> struct ClosedInterval {
	T from;
	T to;
};
template <typename T> inline ClosedInterval<T> operator+ (const T & offset, const ClosedInterval<T> & i) {
	return {offset + i.from, offset + i.to};
}
template <typename T> inline bool contains (const ClosedInterval<T> & i, const T & value) {
	return i.from <= value && value <= i.to;
}

// Type tag to indicate that a value is supposed to be in the non zero domain.
struct PointInNonZeroDomain {
	Point value;
};

// Convolution

// Combinators
template <typename Inner> struct Shifted {
	Point shift;
	Inner inner;

	ClosedInterval<Point> non_zero_domain () const { return shift + inner.non_zero_domain (); }
	int operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		return inner (PointInNonZeroDomain{x.value - shift});
	}
	int operator() (Point x) const { return inner (x - shift); }
};
template <typename Inner> struct Scaled {
	int scale;
	Inner inner;

	ClosedInterval<Point> non_zero_domain () const { return inner.non_zero_domain (); }
	int operator() (PointInNonZeroDomain x) const { return scale * inner (x); }
	int operator() (Point x) const { return scale * inner (x); }
};

// Indicator function for an interval.
struct IntervalIndicator {
	int half_width; // [0, inf[

	ClosedInterval<Point> non_zero_domain () const { return {-half_width, half_width}; }
	int operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		return 1;
	}
	int operator() (Point x) const { return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0; }
};

// Triangle (0,0), (side, 0), (side, side)
struct PositiveTriangle {
	int side; // [0, inf[

	ClosedInterval<Point> non_zero_domain () const { return {0, side}; }
	int operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		return x.value;
	}
	int operator() (Point x) const { return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0; }
};

// Triangle (0,0), (-side, 0), (-side, side)
struct NegativeTriangle {
	int side; // [0, inf[

	ClosedInterval<Point> non_zero_domain () const { return {-side, 0}; }
	int operator() (PointInNonZeroDomain x) const {
		assert (contains (non_zero_domain (), x.value));
		return -x.value;
	}
	int operator() (Point x) const { return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0; }
};

// Trapezoid with a block (2*half_base, height) with a PositiveTriangle(height) on the left and a negative on the right.
struct Trapezoid {
	int height;    // [0, inf[
	int half_base; // [0, inf[

	ClosedInterval<Point> non_zero_domain () const {
		const auto half_len = height + half_base;
		return {-half_len, half_len};
	}
	int operator() (PointInNonZeroDomain x) const {
		if (x.value < -half_base) {
			return x.value - (half_base + height); // Left triangle
		} else if (x.value > half_base) {
			return (half_base + height) - x.value; // Right triangle
		} else {
			return height; // Central block
		}
	}
	int operator() (Point x) const { return contains (non_zero_domain (), x) ? operator() (PointInNonZeroDomain{x}) : 0; }
};

inline Scaled<IntervalIndicator> central_block (const Trapezoid & trapezoid) {
	return {trapezoid.height, {trapezoid.half_base}};
}
inline Shifted<PositiveTriangle> left_triangle (const Trapezoid & trapezoid) {
	return {-(trapezoid.height + trapezoid.half_base), {trapezoid.height}};
}
inline Shifted<NegativeTriangle> right_triangle (const Trapezoid & trapezoid) {
	return {trapezoid.height + trapezoid.half_base, {trapezoid.height}};
}

inline Trapezoid convolution (const IntervalIndicator & left, const IntervalIndicator & right) {
	return {std::min (left.half_width, right.half_width) * 2, std::abs (left.half_width - right.half_width)};
}

} // namespace shape
