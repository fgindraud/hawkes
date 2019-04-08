#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <vector>

#include "utils.h"

// Type tag for "no value"
struct None {};

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

// Interval for a point with uncertainty : center + width of uncertainty
struct PointInterval {
	Point center;
	PointSpace width; // >= 0
};

inline bool operator< (const PointInterval & lhs, const PointInterval & rhs) {
	return lhs.center < rhs.center;
}
inline bool operator== (const PointInterval & lhs, const PointInterval & rhs) {
	return lhs.center == rhs.center && lhs.width == rhs.width;
}

template <typename T> struct DataByProcessRegion {
	Vector2d<T> data_; // Rows = regions, cols = processes.

	DataByProcessRegion (size_t nb_processes, size_t nb_regions) : data_ (nb_regions, nb_processes) {
		assert (nb_processes > 0);
		assert (nb_regions > 0);
	}

	size_t nb_regions () const { return data_.nb_rows (); }
	size_t nb_processes () const { return data_.nb_cols (); }

	const T & data (ProcessId m, RegionId r) const { return data_ (r, m); }
	T & data (ProcessId m, RegionId r) { return data_ (r, m); }
	span<const T> data_for_region (RegionId r) const { return data_.row (r); }
};

/******************************************************************************
 * Function bases.
 */
struct HistogramBase {
	size_t base_size; // [1, inf[
	PointSpace delta; // ]0, inf[

	HistogramBase (size_t base_size, PointSpace delta) : base_size (base_size), delta (delta) {
		assert (base_size > 0);
		assert (delta > 0.);
	}

	// ]left; right]
	struct Interval {
		PointSpace left;
		PointSpace right;
	};

	Interval interval (FunctionBaseId k) const {
		assert (k < base_size);
		return {PointSpace (k) * delta, PointSpace (k + 1) * delta};
	}
};
inline double normalization_factor (HistogramBase base) {
	return 1. / std::sqrt (base.delta);
}

/******************************************************************************
 * Kernels.
 */

// Interval kernel : 1_[-width/2 + center, width/2 + center](x) (L2-normalized)
// 'center' allows this struct to support both centered (center = 0) and uncentered intervals.
struct IntervalKernel {
	PointSpace width; // ]0, inf[ due to the normalization factor
	Point center;

	IntervalKernel (PointSpace width, Point center) : width (width), center (center) { assert (width > 0.); }
	IntervalKernel (PointSpace width) : IntervalKernel (width, 0) {}
};
inline double normalization_factor (IntervalKernel kernel) {
	return 1. / std::sqrt (kernel.width);
}

// Store kernels and maximum width kernels for heterogeneous mode
template <typename T> struct HeterogeneousKernels {
	DataByProcessRegion<std::vector<T>> kernels;
	std::vector<T> maximum_width_kernels; // For each process
};

// Zero width kernels are not supported by computation, replace their width with a 'default' value.
inline PointSpace fix_zero_width (PointSpace width) {
	assert (width >= 0.);
	if (width == 0.) {
		return 1.;
	} else {
		return width;
	}
}

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

	Matrix_M_MK1 (size_t nb_processes, size_t base_size) : nb_processes (nb_processes), base_size (base_size) {
		assert (nb_processes > 0);
		assert (base_size > 0);
		const auto size = 1 + base_size * nb_processes;
		inner = Eigen::MatrixXd::Constant (int(size), int(nb_processes), std::numeric_limits<double>::quiet_NaN ());
	}

	// b_m,0
	double get_0 (ProcessId m) const {
		assert (m < nb_processes);
		return inner (0, int(m));
	}
	void set_0 (ProcessId m, double v) {
		assert (m < nb_processes);
		inner (0, int(m)) = v;
	}

	// b_m,l,k
	int lk_index (ProcessId l, FunctionBaseId k) const {
		assert (l < nb_processes);
		assert (k < base_size);
		return int(1 + l * base_size + k);
	}
	double get_lk (ProcessId m, ProcessId l, FunctionBaseId k) const {
		assert (m < nb_processes);
		return inner (lk_index (l, k), int(m));
	}
	void set_lk (ProcessId m, ProcessId l, FunctionBaseId k, double v) {
		assert (m < nb_processes);
		inner (lk_index (l, k), int(m)) = v;
	}

	// Access vector (m,0) for all m.
	auto m_0_values () const { return inner.row (0); }
	auto m_0_values () { return inner.row (0); }
	// Access sub-matrix (m,{(l,k)}) for all m,l,k
	auto m_lk_values () const { return inner.bottomRows (nb_processes * base_size); }
	auto m_lk_values () { return inner.bottomRows (nb_processes * base_size); }
	// Access vector of (m,{0}U{(l,k)}) for a given m.
	auto values_for_m (ProcessId m) const { return inner.col (m); }
	auto values_for_m (ProcessId m) { return inner.col (m); }
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

	MatrixG (size_t nb_processes, size_t base_size)
	    : nb_processes (nb_processes),
	      base_size (base_size),
	      inner (1 + base_size * nb_processes, 1 + base_size * nb_processes) {
		assert (nb_processes > 0);
		assert (base_size > 0);
		const auto size = 1 + base_size * nb_processes;
		inner = Eigen::MatrixXd::Constant (int(size), size, std::numeric_limits<double>::quiet_NaN ());
	}

	int lk_index (ProcessId l, FunctionBaseId k) const {
		assert (l < nb_processes);
		assert (k < base_size);
		return int(1 + l * base_size + k);
	}

	// Tmax
	double get_tmax () const { return inner (0, 0); }
	void set_tmax (double v) { inner (0, 0) = v; }

	// g_l,k (duplicated with transposition)
	double get_g (ProcessId l, FunctionBaseId k) const { return inner (0, lk_index (l, k)); }
	void set_g (ProcessId l, FunctionBaseId k, double v) {
		const auto i = lk_index (l, k);
		inner (0, i) = inner (i, 0) = v;
	}

	// G_l,l2_k,k2 (symmetric, only need to be set for (l,k) <= [or >=] (l2,k2))
	double get_G (ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) const {
		return inner (lk_index (l, k), lk_index (l2, k2));
	}
	void set_G (ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2, double v) {
		const auto i = lk_index (l, k);
		const auto i2 = lk_index (l2, k2);
		inner (i, i2) = inner (i2, i) = v;
	}
};
