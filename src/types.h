#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <cstdint>
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

// Points are stored as signed integer values to avoid corner cases around the 0 coordinate.
using std::int32_t;
using std::int64_t;

// Single coordinate for a process represented by points. int32 are sufficient (covers +/- 2G).
using Point = int32_t;

// PointSpace is used when any integer is converted to the "point space" before computing with point coordinates.
// This is used to convey the intent more clearly in the code, and ensure safe conversion to signed values for indexes.
using PointSpace = int32_t;

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
	size_t nb_regions () const { return points_.nb_rows (); }
	size_t nb_processes () const { return points_.nb_cols (); }

	const SortedVec<Point> & process_data (ProcessId m, RegionId r) const { return points_ (r, m); }
	span<const SortedVec<Point>> processes_data_for_region (RegionId r) const { return points_.row (r); }

	static ProcessesRegionData from_raw (const std::vector<RawProcessData> & raw_processes) {
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
					const auto point = (interval.left + interval.right) / 2;
					points.emplace_back (point);
				}
				// Apply reversing if requested before sorting them in increasing order
				if (raw_process.direction == RawProcessData::Direction::Backward && !points.empty ()) {
					// Reverse point values, and add the max to have positive positions starting with 0.
					// The shift with max is not necessary as algorithms do no require positive positions in general.
					// TODO remove shifting ?
					const auto max = *std::max_element (points.begin (), points.end ());
					for (auto & point : points) {
						point = max - point;
					}
				}
				data.points_ (r, m) = SortedVec<Point>::from_unsorted (std::move (points));
			}
		}
		return data;
	}
};

/******************************************************************************
 * Function bases.
 */
struct HistogramBase {
	size_t base_size; // [1, inf[
	PointSpace delta; // [1, inf[

	// ]left; right]
	struct Interval {
		PointSpace left;
		PointSpace right;
	};

	Interval interval (FunctionBaseId k) const {
		assert (k < base_size);
		return {PointSpace (k) * delta, (PointSpace (k) + 1) * delta};
	}
};

/******************************************************************************
 * Kernels.
 */

// 1_[-width/2, width/2] (x) * 1/sqrt(width)
struct IntervalKernel {
	PointSpace width;
};

/******************************************************************************
 * Computation matrices.
 */

struct Matrix_M_MK1 {
	// Stores values for a_m_kl, b_m_kl, d_m_kl.
	// Invariant: K > 0 && M > 0 && inner.size() == (1 + K * M, M).
	// Handles conversions of indexes to Eigen indexes (int).

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
	auto m_0_values () const { return inner.row (0); }
	auto m_0_values () { return inner.row (0); }

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
	auto m_lk_values () const { return inner.bottomRows (nb_processes * base_size); }
	auto m_lk_values () { return inner.bottomRows (nb_processes * base_size); }
};

struct MatrixG {
	// Stores value of the G matrix (symmetric).
	// Invariant: K > 0 && M > 0 && inner.size() == (1 + K * M, 1 + K * M)
	// Handles conversions of indexes to Eigen indexes (int).

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
