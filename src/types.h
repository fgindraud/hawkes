#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "utils.h"

using std::int32_t;
using std::int64_t;

/******************************************************************************
 * Index types.
 * TODO homogeneize, use size_t for indexes (IndexType base typedef)
 * TODO use typedef for PointSpaceType = int32_t
 */
struct ProcessId {
	int32_t value; // [0; nb_processes[
};
struct FunctionBaseId {
	int32_t value; // [0; base_size[
};
struct RegionId {
	int32_t value; // [0; nb_regions[
};

/******************************************************************************
 * Process data.
 */

// Single coordinate for a process represented by points.
using Point = int32_t;

// Interval for a point with uncertainty
struct PointInterval {
	Point start;
	Point end;
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
	Vector2d<SortedVec<Point>> points_;

	ProcessesRegionData (std::size_t nb_processes, std::size_t nb_regions) : points_ (nb_regions, nb_processes) {
		assert (nb_processes > 0);
		assert (nb_regions > 0);
	}

public:
	std::size_t nb_regions () const { return points_.nb_rows (); }
	std::size_t nb_processes () const { return points_.nb_cols (); }

	const SortedVec<Point> & process_data (ProcessId m, RegionId r) const { return points_ (r.value, m.value); }
	span<const SortedVec<Point>> processes_data_for_region (RegionId r) const { return points_.row (r.value); }

	static ProcessesRegionData from_raw (const std::vector<RawProcessData> & raw_processes) {
		const auto nb_processes = raw_processes.size ();
		if (raw_processes.empty ()) {
			throw std::runtime_error ("ProcessesRegionData::from_raw: Empty process list");
		}
		const auto nb_regions = raw_processes[0].regions.size ();
		ProcessesRegionData data (nb_processes, nb_regions);
		for (ProcessId m{0}; m.value < nb_processes; m.value++) {
			const auto & raw_process = raw_processes[m.value];
			if (raw_process.regions.size () != nb_regions) {
				throw std::runtime_error (
				    fmt::format ("ProcessesRegionData::from_raw: process {} has wrong region number: got {}, expected {}",
				                 m.value, raw_process.regions.size (), nb_regions));
			}
			for (RegionId r{0}; r.value < nb_regions; ++r.value) {
				// Intervals are represented by their middle points
				std::vector<Point> points;
				points.reserve (raw_process.regions[r.value].unsorted_intervals.size ());
				for (const auto & interval : raw_process.regions[r.value].unsorted_intervals) {
					const auto point = (interval.start + interval.end) / 2;
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
				data.points_ (r.value, m.value) = SortedVec<Point>::from_unsorted (std::move (points));
			}
		}
		return data;
	}
};

/******************************************************************************
 * Function bases.
 */
struct HistogramBase {
	int32_t base_size; // [1, inf[
	int32_t delta;     // [1, inf[

	// ]from; to]
	struct Interval {
		int32_t from;
		int32_t to;
	};

	Interval interval (FunctionBaseId k) const {
		assert (0 <= k.value && k.value < base_size);
		return {k.value * delta, (k.value + 1) * delta};
	}
};

/******************************************************************************
 * Computation matrices.
 */

struct Matrix_M_MK1 {
	// Stores values for a_m_kl, b_m_kl, d_m_kl.
	// Invariant: K > 0 && M > 0 && inner.size() == (1 + K * M, M).

	int32_t nb_processes; // M
	int32_t base_size;    // K
	Eigen::MatrixXd inner;

	Matrix_M_MK1 (int32_t nb_processes, int32_t base_size) : nb_processes (nb_processes), base_size (base_size) {
		assert (nb_processes > 0);
		assert (base_size > 0);
		const auto size = 1 + base_size * nb_processes;
		inner = Eigen::MatrixXd::Constant (size, nb_processes, std::numeric_limits<double>::quiet_NaN ());
	}

	// b_m,0
	double get_0 (ProcessId m) const {
		assert (0 <= m.value && m.value < nb_processes);
		return inner (0, m.value);
	}
	void set_0 (ProcessId m, double v) {
		assert (0 <= m.value && m.value < nb_processes);
		inner (0, m.value) = v;
	}
	auto m_0_values () const { return inner.row (0); }
	auto m_0_values () { return inner.row (0); }

	// b_m,l,k
	int32_t lk_index (ProcessId l, FunctionBaseId k) const {
		assert (0 <= l.value && l.value < nb_processes);
		assert (0 <= k.value && k.value < base_size);
		return 1 + l.value * base_size + k.value;
	}
	double get_lk (ProcessId m, ProcessId l, FunctionBaseId k) const {
		assert (0 <= m.value && m.value < nb_processes);
		return inner (lk_index (l, k), m.value);
	}
	void set_lk (ProcessId m, ProcessId l, FunctionBaseId k, double v) {
		assert (0 <= m.value && m.value < nb_processes);
		inner (lk_index (l, k), m.value) = v;
	}
	auto m_lk_values () const { return inner.bottomRows (nb_processes * base_size); }
	auto m_lk_values () { return inner.bottomRows (nb_processes * base_size); }
};

struct MatrixG {
	// Stores value of the G matrix (symmetric).
	// Invariant: K > 0 && M > 0 && inner.size() == (1 + K * M, 1 + K * M)

	int32_t nb_processes; // M
	int32_t base_size;    // K
	Eigen::MatrixXd inner;

	MatrixG (int32_t nb_processes, int32_t base_size)
	    : nb_processes (nb_processes),
	      base_size (base_size),
	      inner (1 + base_size * nb_processes, 1 + base_size * nb_processes) {
		assert (nb_processes > 0);
		assert (base_size > 0);
		const auto size = 1 + base_size * nb_processes;
		inner = Eigen::MatrixXd::Constant (size, size, std::numeric_limits<double>::quiet_NaN ());
	}

	int32_t lk_index (ProcessId l, FunctionBaseId k) const {
		assert (0 <= l.value && l.value < nb_processes);
		assert (0 <= k.value && k.value < base_size);
		return 1 + l.value * base_size + k.value;
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
