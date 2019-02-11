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

// Interval for a point with uncertainty
struct PointInterval {
	Point left;
	Point right;
};

// Raw process data, read from a file
struct RawRegionData {
	std::string name;
	std::vector<PointInterval> unsorted_intervals;
};
struct RawProcessData {
	std::string name;
	std::vector<RawRegionData> regions;
	enum class Direction { Forward, Backward } direction;
};

// Store the data for multiple processes and regions.
// All processes must have the same number of regions.
// Number of regions and process must be non zero.
class ProcessesRegionData {
private:
	Vector2d<SortedVec<Point>> points_; // Rows = regions, Cols = processes.

	ProcessesRegionData (size_t nb_processes, size_t nb_regions) : points_ (nb_regions, nb_processes) {
		assert (nb_processes > 0);
		assert (nb_regions > 0);
	}

public:
	static ProcessesRegionData from_raw (const std::vector<RawProcessData> & raw_processes);

	size_t nb_regions () const { return points_.nb_rows (); }
	size_t nb_processes () const { return points_.nb_cols (); }

	const SortedVec<Point> & process_data (ProcessId m, RegionId r) const { return points_ (r, m); }
	span<const SortedVec<Point>> processes_data_for_region (RegionId r) const { return points_.row (r); }
};

inline ProcessesRegionData ProcessesRegionData::from_raw (const std::vector<RawProcessData> & raw_processes) {
	const auto nb_processes = raw_processes.size ();
	if (raw_processes.empty ()) {
		throw std::runtime_error ("ProcessesRegionData::from_raw: Empty process list");
	}
	const auto nb_regions = raw_processes[0].regions.size ();
	ProcessesRegionData data (nb_processes, nb_regions);
	for (ProcessId m = 0; m < nb_processes; m++) {
		const auto & raw_process = raw_processes[m];
		if (raw_process.regions.size () != nb_regions) {
			throw std::runtime_error (
			    fmt::format ("ProcessesRegionData::from_raw: process {} has wrong region number: got {}, expected {}", m,
			                 raw_process.regions.size (), nb_regions));
		}
		for (RegionId r = 0; r < nb_regions; ++r) {
			// Intervals are represented by their middle points
			std::vector<Point> points;
			points.reserve (raw_process.regions[r].unsorted_intervals.size ());
			for (const auto & interval : raw_process.regions[r].unsorted_intervals) {
				const auto point = (interval.left + interval.right) / 2.;
				points.emplace_back (point);
			}
			// Apply reversing if requested before sorting them in increasing order
			if (raw_process.direction == RawProcessData::Direction::Backward && !points.empty ()) {
				// Reverse point values
				for (auto & point : points) {
					point = -point;
				}
			}
			data.points_ (r, m) = SortedVec<Point>::from_unsorted (std::move (points));
		}
	}
	return data;
}

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

// 1_[-width/2, width/2]
struct IntervalKernel {
	PointSpace width; // ]0, inf[ due to the normalization factor

	IntervalKernel (PointSpace width) : width (width) { assert (width > 0.); }
};
inline double normalization_factor (IntervalKernel kernel) {
	return 1. / std::sqrt (kernel.width);
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

/******************************************************************************
 * FIXME Experimental: interval kernel specific by point.
 */
struct PointAndKernel {
	Point point;
	IntervalKernel kernel;
};
class PointAndKernelData {
private:
	Vector2d<std::vector<PointAndKernel>> data;

	PointAndKernelData (size_t nb_processes, size_t nb_regions) : data (nb_regions, nb_processes) {
		assert (nb_processes > 0);
		assert (nb_regions > 0);
	}

public:
	static PointAndKernelData from_raw (const std::vector<RawProcessData> & raw_processes);

	size_t nb_regions () const { return data.nb_rows (); }
	size_t nb_processes () const { return data.nb_cols (); }

	const std::vector<PointAndKernel> & process_data (ProcessId m, RegionId r) const { return data (r, m); }
	span<const std::vector<PointAndKernel>> processes_data_for_region (RegionId r) const { return data.row (r); }
};
inline PointAndKernelData PointAndKernelData::from_raw (const std::vector<RawProcessData> & raw_processes) {
	const auto nb_processes = raw_processes.size ();
	if (raw_processes.empty ()) {
		throw std::runtime_error ("PointAndKernelData::from_raw: Empty process list");
	}
	const auto nb_regions = raw_processes[0].regions.size ();
	PointAndKernelData data (nb_processes, nb_regions);
	for (ProcessId m = 0; m < nb_processes; m++) {
		const auto & raw_process = raw_processes[m];
		if (raw_process.regions.size () != nb_regions) {
			throw std::runtime_error (
			    fmt::format ("PointAndKernelData::from_raw: process {} has wrong region number: got {}, expected {}", m,
			                 raw_process.regions.size (), nb_regions));
		}
		for (RegionId r = 0; r < nb_regions; ++r) {
			// Intervals are represented by their middle points
			std::vector<PointAndKernel> points;
			points.reserve (raw_process.regions[r].unsorted_intervals.size ());
			for (const auto & interval : raw_process.regions[r].unsorted_intervals) {
				const auto point = (interval.left + interval.right) / 2.;
				const auto kernel = IntervalKernel (std::max (interval.right - interval.left, 1.));
				points.emplace_back (PointAndKernel{point, kernel});
			}
			// Apply reversing if requested before sorting them in increasing order
			if (raw_process.direction == RawProcessData::Direction::Backward && !points.empty ()) {
				// Reverse point values
				for (auto & point : points) {
					point.point = -point.point;
				}
			}
			data.data (r, m) = std::move (points);
		}
	}
	return data;
}
