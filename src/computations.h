#pragma once

#include <cmath>
#include <limits>

#include "types.h"

inline std::int64_t tmax_for_region (const ProcessesData<Point> & processes, RegionId region) {
	std::int64_t min = std::numeric_limits<std::int64_t>::max ();
	std::int64_t max = std::numeric_limits<std::int64_t>::min ();

	for (ProcessId m{0}; m.value < processes.nb_processes (); ++m.value) {
		const auto & points = processes.data (m, region);
		if (points.size () > 0) {
			min = std::min (min, std::int64_t (points[0]));
			max = std::max (max, std::int64_t (points[points.size () - 1]));
		}
	}

	if (min <= max) {
		return max - min;
	} else {
		return 0; // If there are no points at all, return 0
	}
}

inline std::int64_t count_point_difference_in_interval (const SortedVec<Point> & m_process,
                                                        const SortedVec<Point> & l_process,
                                                        HistogramBase::Interval interval) {
	// Count pair of points within the k-th interval
	// TODO impl by doing all k at once, from rust experiment
	const auto n_m = m_process.size ();
	std::int64_t count = 0;
	int last_starting_i = 0;
	for (const Point x_l : l_process) {
		// Count x_m in ]x_l + interval.from, x_l + interval.to]

		// Find first i where Nm[i] is in the interval
		while (last_starting_i < n_m && !(x_l + interval.from < m_process[last_starting_i])) {
			last_starting_i += 1;
		}
		// i is out of bounds or Nm[i] > x_l + interval.from
		int i = last_starting_i;
		// Count elements of Nm still in interval
		while (i < n_m && m_process[i] <= x_l + interval.to) {
			count += 1;
			i += 1;
		}
	}
	return count;
}

inline std::int64_t compute_cross_correlation (const SortedVec<Point> & l_process, const SortedVec<Point> & l2_process,
                                               HistogramBase::Interval interval, HistogramBase::Interval interval2) {
	constexpr Point inf = std::numeric_limits<Point>::max ();

	struct SlidingInterval {
		const SortedVec<Point> & points; // Points to slide on
		HistogramBase::Interval shifts;  // Shifting of interval bounds

		int current_points_inside = 0;
		// Indexes of next points to enter/exit interval
		int next_i_entering = 0;
		int next_i_exiting = 0;
		// Value of next points to enter/exit interval, or inf if no more points
		Point next_x_entering = 0;
		Point next_x_exiting = 0;

		SlidingInterval (const SortedVec<Point> & points, HistogramBase::Interval shifts)
		    : points (points), shifts (shifts) {
			next_x_entering = get_shifted_point (0, shifts.from);
			next_x_exiting = get_shifted_point (0, shifts.to);
		}
		Point get_shifted_point (int i, int shift) const {
			if (i < points.size ()) {
				return points[i] + shift;
			} else {
				return inf;
			}
		}
		void advance_to (Point new_x) {
			if (new_x == next_x_entering) {
				current_points_inside += 1;
				next_i_entering += 1;
				next_x_entering = get_shifted_point (next_i_entering, shifts.from);
			}
			if (new_x == next_x_exiting) {
				current_points_inside += 1;
				next_i_exiting += 1;
				next_x_exiting = get_shifted_point (next_i_exiting, shifts.to);
			}
		}
	};

	std::int64_t accumulated_area = 0;
	Point current_x = 0;
	SlidingInterval si1 (l_process, interval);
	SlidingInterval si2 (l2_process, interval2);

	while (true) {
		const Point next_x = std::min ({si1.next_x_entering, si1.next_x_exiting, si2.next_x_entering, si2.next_x_exiting});
		if (next_x == inf) {
			// No more points to process, all next points are inf.
			assert (si1.current_points_inside == 0);
			assert (si2.current_points_inside == 0);
			break;
		}
		// Integrate the constant between current and next x.
		accumulated_area += si1.current_points_inside * si2.current_points_inside * (next_x - current_x);
		// Move reference to next_x
		current_x = next_x;
		si1.advance_to (next_x);
		si2.advance_to (next_x);
		// TODO improvement: when one of SI.current_points_inside is 0, advance to next entering of this SI
	}
	return accumulated_area;
}

inline MatrixB compute_b (const ProcessesData<Point> & processes, RegionId region, const HistogramBase & base) {
	const auto nb_processes = processes.nb_processes ();
	const auto base_size = base.base_size;
	MatrixB b (nb_processes, base_size);

	for (ProcessId m{0}; m.value < nb_processes; ++m.value) {
		const auto & m_process = processes.data (m, region);

		// b0
		b.set_0 (m, double(m_process.size ()));

		// b_lk
		for (ProcessId l{0}; l.value < nb_processes; ++l.value) {
			for (FunctionBaseId k{0}; k.value < base_size; ++k.value) {
				const auto count =
				    count_point_difference_in_interval (m_process, processes.data (l, region), base.interval (k));
				b.set_lk (m, l, k, double(count));
			}
		}
	}
	return b;
}

inline MatrixG compute_g (const ProcessesData<Point> & processes, RegionId region, const HistogramBase & base) {
	const auto nb_processes = processes.nb_processes ();
	const auto base_size = base.base_size;
	MatrixG g (nb_processes, base_size);

	g.set_tmax (tmax_for_region (processes, region));

	for (ProcessId l{0}; l.value < nb_processes; ++l.value) {
		const auto g_lk = processes.data (l, region).size () * std::sqrt (base.delta);
		for (FunctionBaseId k{0}; k.value < base_size; ++k.value) {
			g.set_g (l, k, g_lk);
		}
	}

	// G symmetric, only compute for (l2,k2) >= (l,k)
	for (ProcessId l{0}; l.value < nb_processes; ++l.value) {
		for (ProcessId l2{l.value}; l2.value < nb_processes; ++l2.value) {
			for (FunctionBaseId k{0}; k.value < base_size; ++k.value) {
				for (FunctionBaseId k2{k.value}; k2.value < base_size; ++k2.value) {
					const auto v = compute_cross_correlation (processes.data (l, region), processes.data (l2, region),
					                                          base.interval (k), base.interval (k2));
					g.set_G (l, l2, k, k2, double(v));
				}
			}
		}
	}
	return g;
}
