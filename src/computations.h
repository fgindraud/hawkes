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

/* Compute b_{m,l,k} for all k, for the histogram base.
 * Return a vector with the k values.
 * Complexity is O( |N_l| + (K+1) * |N_m| ).
 *
 * In the histogram case, b_{m,l,k} = sum_{(x_l,x_m) in (N_l,N_m) / k*delta < x_m - x_l <= (k+1)*delta} 1.
 * The strategy is to count points of N_l in the interval ] k*delta + x_l, (k+1)*delta + x_l ] for each x_l.
 * This specific functions does it for all k at once.
 * This is more efficient because the upper bound of the k-th interval is the lower bound of the (k+1)-th.
 * Thus we compute the bounds only once.
 */
inline std::vector<std::int64_t> compute_b_ml_histogram_for_all_k (const SortedVec<Point> & m_process,
                                                                   const SortedVec<Point> & l_process,
                                                                   const HistogramBase & base) {
	// Accumulator for sum_{x_l} count ({x_m, (x_m - x_l) in ] k*delta, (k+1)*delta ]})
	std::vector<std::int64_t> counts (base.base_size, 0);
	// Invariant: for x_l last visited point of process l:
	// sib[k] = index of first x_m with x_m - x_l > k*delta
	std::vector<int> sliding_interval_bounds (base.base_size + 1, 0);

	const auto n_m = m_process.size ();
	for (const Point x_l : l_process) {
		// Compute indexes of all k interval boundaries, shifted from the current x_l.
		// This can be done by searching m points starting at the previous positions (x_l increased).
		for (int k = 0; k < base.base_size + 1; ++k) {
			const int shift = k * base.delta;
			int i = sliding_interval_bounds[k];
			while (i < n_m && !(m_process[i] - x_l > shift)) {
				i += 1;
			}
			sliding_interval_bounds[k] = i;
		}
		// Accumulate the number of points in each shifted interval for the current x_l.
		// Number of points = difference between indexes of interval boundaries.
		for (int k = 0; k < base.base_size; ++k) {
			counts[k] += sliding_interval_bounds[k + 1] - sliding_interval_bounds[k];
		}
	}
	return counts;
}

/* Compute G_{l,l2,k,k2} in the histogram case.
 * Complexity is O( |N_l| + |N_m| ).
 *
 * G_{l,l2,k,k2} = integral_x sum_{x_l,x_l2} phi_k (x - x_l) phi_k2 (x - x_l2) dx.
 * G_{l,l2,k,k2} = integral_x N_l(]x - (k+1)*delta, x - k*delta]) N_l2(]x - (k2+1)*delta, x - k2*delta]) dx.
 * With N_l(I) = sum_{x_l in N_l / x_l in I} 1 = number of points of N_l in interval I.
 * This product of counts is constant by chunks.
 * The strategy is to compute the integral by splitting R in the constant parts of N_l(..) * N_l2(..).
 * Thus we loop over all points of changes of this product.
 */
inline std::int64_t compute_g_ll2kk2_histogram (const SortedVec<Point> & l_process, const SortedVec<Point> & l2_process,
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
				current_points_inside -= 1;
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
			auto counts = compute_b_ml_histogram_for_all_k (m_process, processes.data (l, region), base);
			for (FunctionBaseId k{0}; k.value < base_size; ++k.value) {
				b.set_lk (m, l, k, double(counts[k.value]));
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
					const auto v = compute_g_ll2kk2_histogram (processes.data (l, region), processes.data (l2, region),
					                                           base.interval (k), base.interval (k2));
					g.set_G (l, l2, k, k2, double(v));
				}
			}
		}
	}
	return g;
}
