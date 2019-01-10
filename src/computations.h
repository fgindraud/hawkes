#pragma once

#include <cmath>
#include <limits>

#include "convolution.h"
#include "types.h"

/******************************************************************************
 * Generic functions useful for all cases.
 */

/* Compute sum_{x_m in N_m, x_l in N_l} shape(x_m - x_l).
 * Shape must be any shape from the shape namespace:
 * - with a method non_zero_domain() returning the interval where the shape is non zero.
 * - with an operator()(x) returning the value at point x.
 *
 * Worst case complexity: O(|N|^2).
 * Average complexity: O(|N| * density(N) * width(shape)) = O(|N|^2 * width(shape) / Tmax).
 */
template <typename Shape>
inline auto compute_sum_of_point_differences (const SortedVec<Point> & m_points, const SortedVec<Point> & l_points,
                                              const Shape & shape) {
	using ReturnType = decltype (shape (Point{}));
	ReturnType sum{};

	// shape(x) != 0 => x in shape.non_zero_domain().
	// Thus sum_{x_m,x_l} shape(x_m - x_l) = sum_{(x_m, x_l), x_m - x_l in non_zero_domain} shape(x_m - x_l).
	const auto non_zero_domain = shape.non_zero_domain ();

	int32_t starting_i_m = 0;
	for (const Point x_l : l_points) {
		// x_l = N_l[i_l], with N_l[x] a strictly increasing function of x.
		// Compute shape(x_m - x_l) for all x_m in (x_l + non_zero_domain) interval.
		const auto interval_i_l = x_l + non_zero_domain;

		// starting_i_m = min{i_m, N_m[i_m] - N_l[i_l] >= non_zero_domain.from}.
		// We can restrict the search by starting from:
		// last_starting_i_m = min{i_m, N_m[i_m] - N_l[i_l - 1] >= non_zero_domain.from or i_m == 0}.
		// We have: N_m[starting_i_m] >= N_l[i_l] + nzd.from > N_l[i_l - 1] + nzd.from.
		// Because N_m is increasing and properties of the min, starting_i_m >= last_starting_i_m.
		while (starting_i_m < m_points.size () && !(interval_i_l.from <= m_points[starting_i_m])) {
			starting_i_m += 1;
		}
		if (starting_i_m == m_points.size ()) {
			// starting_i_m is undefined because last(N_m) < N_l[i_l] + non_zero_domain.from.
			// last(N_m) == max(x_m in N_m) because N_m[x] is strictly increasing.
			// So for each j > i_l , max(x_m) < N[j] + non_zero_domain.from, and shape (x_m - N_l[j]) == 0.
			// We can stop there as the sum is already complete.
			break;
		}
		// Sum values of shape(x_m - x_l) as long as x_m is in interval_i_l.
		// starting_i_m defined => for each i_m < starting_i_m, shape(N_m[i_m] - x_l) == 0.
		// Thus we only scan from starting_i_m to the last i_m in interval.
		// N_m[x] is strictly increasing so we only need to check the right bound of the interval.
		for (int32_t i_m = starting_i_m; i_m < m_points.size () && m_points[i_m] <= interval_i_l.to; i_m += 1) {
			sum += shape (shape::PointInNonZeroDomain{m_points[i_m] - x_l});
		}
	}

	return sum;
}

// Scaling can be moved out of computation.
template <typename T, typename Inner>
inline auto compute_sum_of_point_differences (const SortedVec<Point> & m_points, const SortedVec<Point> & l_points,
                                              const shape::Scaled<T, Inner> & shape) {
	return shape.scale * compute_sum_of_point_differences (m_points, l_points, shape.inner);
}

/* Compute Tmax (used in G).
 */
inline int64_t tmax (span<const SortedVec<Point>> processes) {
	int64_t min = std::numeric_limits<int64_t>::max ();
	int64_t max = std::numeric_limits<int64_t>::min ();

	for (const auto & points : processes) {
		if (points.size () > 0) {
			min = std::min (min, int64_t (points[0]));
			max = std::max (max, int64_t (points[points.size () - 1]));
		}
	}

	if (min <= max) {
		return max - min;
	} else {
		return 0; // If there are no points at all, return 0
	}
}

/******************************************************************************
 * Basic histogram case.
 */

/* Compute b_{m,l,k} * sqrt(delta) for all k, for the histogram base.
 * Return a vector with the k values.
 * Complexity is O( |N_l| + (K+1) * |N_m| ).
 *
 * In the histogram case, b_{m,l,k} = sum_{(x_l,x_m) in (N_l,N_m) / k*delta < x_m - x_l <= (k+1)*delta} 1/sqrt(delta).
 * The strategy is to count points of N_l in the interval ] k*delta + x_l, (k+1)*delta + x_l ] for each x_l.
 * This specific functions does it for all k at once.
 * This is more efficient because the upper bound of the k-th interval is the lower bound of the (k+1)-th.
 * Thus we compute the bounds only once.
 */
inline std::vector<int64_t> compute_b_ml_histogram_counts_for_all_k (const SortedVec<Point> & m_points,
                                                                     const SortedVec<Point> & l_points,
                                                                     const HistogramBase & base) {
	// Accumulator for sum_{x_l} count ({x_m, (x_m - x_l) in ] k*delta, (k+1)*delta ]})
	std::vector<int64_t> counts (base.base_size, 0);
	// Invariant: for x_l last visited point of process l:
	// sib[k] = index of first x_m with x_m - x_l > k*delta
	std::vector<int32_t> sliding_interval_bounds (base.base_size + 1, 0);

	const auto n_m = m_points.size ();
	for (const Point x_l : l_points) {
		// Compute indexes of all k interval boundaries, shifted from the current x_l.
		// This can be done by searching m points starting at the previous positions (x_l increased).
		for (int32_t k = 0; k < base.base_size + 1; ++k) {
			const int32_t shift = k * base.delta;
			auto i = sliding_interval_bounds[k];
			while (i < n_m && !(m_points[i] - x_l > shift)) {
				i += 1;
			}
			sliding_interval_bounds[k] = i;
		}
		// Accumulate the number of points in each shifted interval for the current x_l.
		// Number of points = difference between indexes of interval boundaries.
		for (int32_t k = 0; k < base.base_size; ++k) {
			counts[k] += sliding_interval_bounds[k + 1] - sliding_interval_bounds[k];
		}
	}
	return counts;
}

/* Compute G_{l,l2,k,k2} * delta in the histogram case.
 * Complexity is O( |N_l| + |N_m| ).
 *
 * G_{l,l2,k,k2} = integral_x sum_{x_l,x_l2} phi_k (x - x_l) phi_k2 (x - x_l2) dx.
 * G_{l,l2,k,k2}*delta = integral_x N_l(]x - (k+1)*delta, x - k*delta]) N_l2(]x - (k2+1)*delta, x - k2*delta]) dx.
 * With N_l(I) = sum_{x_l in N_l / x_l in I} 1 = number of points of N_l in interval I.
 * This product of counts is constant by chunks.
 * The strategy is to compute the integral by splitting R in the constant parts of N_l(..) * N_l2(..).
 * Thus we loop over all points of changes of this product.
 */
inline int64_t compute_g_ll2kk2_histogram_integral (const SortedVec<Point> & l_points,
                                                    const SortedVec<Point> & l2_points,
                                                    HistogramBase::Interval interval,
                                                    HistogramBase::Interval interval2) {
	// TODO replace Point by int64_t to avoid overflows ?
	constexpr Point inf = std::numeric_limits<Point>::max ();

	struct SlidingInterval {
		const SortedVec<Point> & points; // Points to slide on
		HistogramBase::Interval shifts;  // Shifting of interval bounds

		int64_t current_points_inside = 0;
		// Indexes of next points to enter/exit interval
		int32_t next_i_entering = 0;
		int32_t next_i_exiting = 0;
		// Value of next points to enter/exit interval, or inf if no more points
		Point next_x_entering = 0;
		Point next_x_exiting = 0;

		SlidingInterval (const SortedVec<Point> & points, HistogramBase::Interval shifts)
		    : points (points), shifts (shifts) {
			next_x_entering = get_shifted_point (0, shifts.from);
			next_x_exiting = get_shifted_point (0, shifts.to);
		}
		Point get_shifted_point (int32_t i, int32_t shift) const {
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

	int64_t accumulated_area = 0;
	Point current_x = 0;
	SlidingInterval si1 (l_points, interval);
	SlidingInterval si2 (l2_points, interval2);

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

// TODO: B and G are sums of region-specific B and G !

// Complexity: O( M^2 * K * max(|N_m|) ).
inline Matrix_M_MK1 compute_b (span<const SortedVec<Point>> processes, HistogramBase base) {
	const auto nb_processes = int32_t (processes.size ());
	const auto base_size = base.base_size;
	const auto inv_sqrt_delta = 1. / std::sqrt (double(base.delta));
	Matrix_M_MK1 b (nb_processes, base_size);

	for (ProcessId m{0}; m.value < nb_processes; ++m.value) {
		const auto & m_process = processes[m.value];

		// b_0
		b.set_0 (m, double(m_process.size ()));

		// b_lk
		for (ProcessId l{0}; l.value < nb_processes; ++l.value) {
			auto counts = compute_b_ml_histogram_counts_for_all_k (m_process, processes[l.value], base);
			for (FunctionBaseId k{0}; k.value < base_size; ++k.value) {
				b.set_lk (m, l, k, double(counts[k.value]) * inv_sqrt_delta);
			}
		}
	}
	return b;
}

// Complexity: O( M^2 * K * max(|N_m|) ).
inline MatrixG compute_g (span<const SortedVec<Point>> processes, HistogramBase base) {
	const auto nb_processes = int32_t (processes.size ());
	const auto base_size = base.base_size;
	const auto sqrt_delta = std::sqrt (double(base.delta));
	const auto inv_delta = 1 / double(base.delta);
	MatrixG g (nb_processes, base_size);

	g.set_tmax (tmax (processes));

	for (ProcessId l{0}; l.value < nb_processes; ++l.value) {
		const auto g_lk = processes[l.value].size () * sqrt_delta;
		for (FunctionBaseId k{0}; k.value < base_size; ++k.value) {
			g.set_g (l, k, g_lk);
		}
	}

	auto G_value = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
		const auto integral = compute_g_ll2kk2_histogram_integral (processes[l.value], processes[l2.value],
		                                                           base.interval (k), base.interval (k2));
		return double(integral) * inv_delta;
	};
	/* G symmetric, only compute for (l2,k2) >= (l,k) (lexicographically).
	 *
	 * G_{l,l2,k,k2} = integral_x sum_{x_l,x_l2} phi_k (x - x_l) phi_k2 (x - x_l2) dx.
	 * By change of variable x -> x + c*delta: G_{l,l2,k,k2} = G_{l,l2,k+c,k2+c}.
	 * Thus for each block of G for (l,l2), we only need to compute 2K+1 values of G.
	 * We compute the values on two borders, and copy the values for all inner cells.
	 */
	for (ProcessId l{0}; l.value < nb_processes; ++l.value) {
		// Case l2 == l: compute for k2 >= k:
		for (FunctionBaseId k{0}; k.value < base_size; ++k.value) {
			// Compute G_{l,l,0,k} and copy to G_{l,l,c,k+c} for k in [0,K[.
			const auto v = G_value (l, l, FunctionBaseId{0}, k);
			for (int32_t c = 0; k.value + c < base_size; ++c) {
				g.set_G (l, l, FunctionBaseId{c}, FunctionBaseId{k.value + c}, v);
			}
		}
		// Case l2 > l: compute for all (k,k2):
		for (ProcessId l2{l.value}; l2.value < nb_processes; ++l2.value) {
			// Compute G_{l,l2,0,0} and copy to G_{l,l2,c,c}.
			{
				const auto v = G_value (l, l2, FunctionBaseId{0}, FunctionBaseId{0});
				for (int32_t c = 0; c < base_size; ++c) {
					g.set_G (l, l2, FunctionBaseId{c}, FunctionBaseId{c}, v);
				}
			}
			/* for k in [1,K[:
			 * Compute G_{l,l2,0,k} and copy to G_{l,l2,c,k+c}.
			 * Compute G_{l,l2,k,0} and copy to G_{l,l2,k+c,c}.
			 */
			for (FunctionBaseId k{1}; k.value < base_size; ++k.value) {
				const auto v_0k = G_value (l, l2, FunctionBaseId{0}, k);
				const auto v_k0 = G_value (l, l2, k, FunctionBaseId{0});
				for (int32_t c = 0; k.value + c < base_size; ++c) {
					g.set_G (l, l2, FunctionBaseId{c}, FunctionBaseId{k.value + c}, v_0k);
					g.set_G (l, l2, FunctionBaseId{k.value + c}, FunctionBaseId{c}, v_k0);
				}
			}
		}
	}
	return g;
}

inline Matrix_M_MK1 compute_d (double gamma, span<const Matrix_M_MK1> b_by_region) {
	const auto nb_regions = int32_t (b_by_region.size ());
	assert (nb_regions > 0);
	const auto nb_processes = b_by_region[0].nb_processes;
	const auto base_size = b_by_region[0].base_size;

	// Compute V_hat_m_kl * R^2. Division by R^2 is done later.
	Matrix_M_MK1 v_hat (nb_processes, base_size);
	for (const auto & b : b_by_region) {
		v_hat.m_lk_values ().array () += b.m_lk_values ().array ().square ();
	}
	const auto v_hat_factor = 1. / double(nb_regions * nb_regions) * 2 * gamma *
	                          std::log (nb_processes + nb_processes * nb_processes * base_size);

	// Compute B_hat_m_kl

	Matrix_M_MK1 d (nb_processes, base_size);
	d.m_0_values ().setZero (); // No penalty for constant component of estimators
	d.m_lk_values () = (v_hat_factor * v_hat.m_lk_values ().array ()).sqrt ();
	return d;
}

/******************************************************************************
 * Histogram with interval convolution kernels.
 */

// W_width
struct IntervalKernel {
	int32_t width;
};

inline auto to_shape (HistogramBase::Interval i) {
	// TODO Histo::Interval is ]from; to], but shape::Interval is [from; to].
	const auto delta = i.to - i.from;
	const auto center = (i.from + i.to) / 2;
	return shape::scaled (1. / std::sqrt (delta), shape::shifted (center, shape::IntervalIndicator::with_width (delta)));
}
inline auto to_shape (IntervalKernel kernel) {
	return shape::scaled (1. / std::sqrt (kernel.width), shape::IntervalIndicator::with_width (kernel.width));
}

inline double compute_b_mlk_histogram (const SortedVec<Point> & m_points, const SortedVec<Point> & l_points,
                                       HistogramBase::Interval base_interval, IntervalKernel m_kernel,
                                       IntervalKernel l_kernel) {
	// Base shapes
	const auto w_m = to_shape (m_kernel);
	const auto w_l = to_shape (l_kernel);
	const auto phi_k = to_shape (base_interval);
	// To compute b_mlk, we need to compute convolution(w_m,w_l,phi_k)(x):
	const auto trapezoid = convolution (w_m, w_l);
	const auto trapezoid_l = component (trapezoid, shape::Trapezoid::LeftTriangle{});
	const auto trapezoid_c = component (trapezoid, shape::Trapezoid::CentralBlock{});
	const auto trapezoid_r = component (trapezoid, shape::Trapezoid::RightTriangle{});
	// b_mlk = sum_{x_m in N_m, x_l in N_l} convolution(w_m,w_l,phi_k)(x_m - x_l)
	// Split the sum into separate sums for each component.
	const auto sum_value_for = [&m_points, &l_points](const auto & shape) {
		return compute_sum_of_point_differences (m_points, l_points, shape);
	};
	return sum_value_for (convolution (trapezoid_l, phi_k)) + sum_value_for (convolution (trapezoid_c, phi_k)) +
	       sum_value_for (convolution (trapezoid_r, phi_k));
}

inline double compute_g_ll2kk2_histogram_integral (const SortedVec<Point> & l_points,
                                                   const SortedVec<Point> & l2_points, int32_t delta, FunctionBaseId k,
                                                   FunctionBaseId k2, IntervalKernel kernel, IntervalKernel kernel2) {
	// V = sum_{x_l,x_l2} corr(conv(W_l,phi_k),conv(W_l2,phi_k2)) (x_l-x_l2)
	// V = factorized_scaling * sum_{x_l,x_l2} shifted((k2-k)*delta, conv(trapezoid(l), trapezoid(l2))) (x_l-x_l2)

	// prepare common scaling and shift
	const auto factorized_scaling = 1. / (double(delta) * std::sqrt (kernel.width * kernel2.width));
	const auto factorized_shift = delta * (k2.value - k.value);
	// compute the two sub-convolutions shapes, shift one of them
	const auto phi_indicator = shape::IntervalIndicator::with_width (delta);
	const auto kernel_indicator = shape::IntervalIndicator::with_width (kernel.width);
	const auto kernel2_indicator = shape::IntervalIndicator::with_width (kernel2.width);
	const auto trapezoid = convolution (kernel_indicator, phi_indicator);
	const auto trapezoid2 = convolution (kernel2_indicator, phi_indicator);
	const auto s_trapezoid = shape::shifted (factorized_shift, trapezoid);
	// decompose
	const auto trapezoid_l = component (s_trapezoid, shape::Trapezoid::LeftTriangle{});
	const auto trapezoid_c = component (s_trapezoid, shape::Trapezoid::CentralBlock{});
	const auto trapezoid_r = component (s_trapezoid, shape::Trapezoid::RightTriangle{});
	const auto trapezoid2_l = component (trapezoid2, shape::Trapezoid::LeftTriangle{});
	const auto trapezoid2_c = component (trapezoid2, shape::Trapezoid::CentralBlock{});
	const auto trapezoid2_r = component (trapezoid2, shape::Trapezoid::RightTriangle{});
	// compute values for each individual convolution shape
	const auto sum_value_for = [&l_points, &l2_points](const auto & shape) {
		return compute_sum_of_point_differences (l_points, l2_points, shape);
	};
	const auto unscaled_sum = sum_value_for (convolution (trapezoid_l, trapezoid2_l)) +
	                          sum_value_for (convolution (trapezoid_l, trapezoid2_c)) +
	                          sum_value_for (convolution (trapezoid_l, trapezoid2_r)) +
	                          sum_value_for (convolution (trapezoid_c, trapezoid2_l)) +
	                          sum_value_for (convolution (trapezoid_c, trapezoid2_c)) +
	                          sum_value_for (convolution (trapezoid_c, trapezoid2_r)) +
	                          sum_value_for (convolution (trapezoid_r, trapezoid2_l)) +
	                          sum_value_for (convolution (trapezoid_r, trapezoid2_c)) +
	                          sum_value_for (convolution (trapezoid_r, trapezoid2_r));
	return factorized_scaling * unscaled_sum;
}

inline Matrix_M_MK1 compute_b (span<const SortedVec<Point>> processes, HistogramBase base,
                               span<const IntervalKernel> kernels) {
	assert (kernels.size () == processes.size ());
	const auto nb_processes = int32_t (processes.size ());
	const auto base_size = base.base_size;
	Matrix_M_MK1 b (nb_processes, base_size);

	for (ProcessId m{0}; m.value < nb_processes; ++m.value) {
		const auto & m_process = processes[m.value];
		const auto & m_kernel = kernels[m.value];

		// b0
		b.set_0 (m, double(m_process.size ()) * std::sqrt (m_kernel.width));

		// b_lk
		for (ProcessId l{0}; l.value < nb_processes; ++l.value) {
			const auto & l_kernel = kernels[l.value];
			for (FunctionBaseId k{0}; k.value < base_size; ++k.value) {
				const auto v = compute_b_mlk_histogram (m_process, processes[l.value], base.interval (k), m_kernel, l_kernel);
				b.set_lk (m, l, k, v);
			}
		}
	}
	return b;
}

inline MatrixG compute_g (span<const SortedVec<Point>> processes, HistogramBase base,
                          span<const IntervalKernel> kernels) {
	assert (kernels.size () == processes.size ());
	const auto nb_processes = int32_t (processes.size ());
	const auto base_size = base.base_size;
	const auto sqrt_delta = std::sqrt (double(base.delta));
	MatrixG g (nb_processes, base_size);

	g.set_tmax (tmax (processes));

	for (ProcessId l{0}; l.value < nb_processes; ++l.value) {
		/* g_lk = sum_{x_m} integral convolution(w_l,phi_k) (x - x_m) dx.
		 * g_lk = sum_{x_m} (integral w_l) (integral phi_k) = sum_{x_m} eta_l sqrt(delta) = |N_m| eta_l sqrt(delta).
		 */
		const auto g_lk = double(processes[l.value].size ()) * std::sqrt (kernels[l.value].width) * sqrt_delta;
		for (FunctionBaseId k{0}; k.value < base_size; ++k.value) {
			g.set_g (l, k, g_lk);
		}
	}

	auto G_value = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
		return compute_g_ll2kk2_histogram_integral (processes[l.value], processes[l2.value], base.delta, k, k2,
		                                            kernels[l.value], kernels[l2.value]);
	};
	/* G symmetric, only compute for (l2,k2) >= (l,k) (lexicographically).
	 *
	 * G_{l,l2,k,k2} = integral_x sum_{x_l,x_l2} conv(w_l,phi_k) (x - x_l) conv(w_l2,phi_k2) (x - x_l2) dx.
	 * By change of variable x -> x + c*delta: G_{l,l2,k,k2} = G_{l,l2,k+c,k2+c}.
	 * Thus for each block of G for (l,l2), we only need to compute 2K+1 values of G.
	 * We compute the values on two borders, and copy the values for all inner cells.
	 */
	for (ProcessId l{0}; l.value < nb_processes; ++l.value) {
		// Case l2 == l: compute for k2 >= k:
		for (FunctionBaseId k{0}; k.value < base_size; ++k.value) {
			// Compute G_{l,l,0,k} and copy to G_{l,l,c,k+c} for k in [0,K[.
			const auto v = G_value (l, l, FunctionBaseId{0}, k);
			for (int32_t c = 0; k.value + c < base_size; ++c) {
				g.set_G (l, l, FunctionBaseId{c}, FunctionBaseId{k.value + c}, v);
			}
		}
		// Case l2 > l: compute for all (k,k2):
		for (ProcessId l2{l.value}; l2.value < nb_processes; ++l2.value) {
			// Compute G_{l,l2,0,0} and copy to G_{l,l2,c,c}.
			{
				const auto v = G_value (l, l2, FunctionBaseId{0}, FunctionBaseId{0});
				for (int32_t c = 0; c < base_size; ++c) {
					g.set_G (l, l2, FunctionBaseId{c}, FunctionBaseId{c}, v);
				}
			}
			/* for k in [1,K[:
			 * Compute G_{l,l2,0,k} and copy to G_{l,l2,c,k+c}.
			 * Compute G_{l,l2,k,0} and copy to G_{l,l2,k+c,c}.
			 */
			for (FunctionBaseId k{1}; k.value < base_size; ++k.value) {
				const auto v_0k = G_value (l, l2, FunctionBaseId{0}, k);
				const auto v_k0 = G_value (l, l2, k, FunctionBaseId{0});
				for (int32_t c = 0; k.value + c < base_size; ++c) {
					g.set_G (l, l2, FunctionBaseId{c}, FunctionBaseId{k.value + c}, v_0k);
					g.set_G (l, l2, FunctionBaseId{k.value + c}, FunctionBaseId{c}, v_k0);
				}
			}
		}
	}
	return g;
}
