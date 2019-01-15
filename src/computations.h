#pragma once

#include <cmath>
#include <limits>

#include "shape.h"
#include "types.h"

/******************************************************************************
 * Generic functions useful for all cases.
 */

/* Compute Tmax (used in G).
 * Tmax is the maximum width covered by points of all processes for a specific region.
 */
inline PointSpace tmax (span<const SortedVec<Point>> processes) {
	PointSpace min = std::numeric_limits<PointSpace>::max ();
	PointSpace max = std::numeric_limits<PointSpace>::min ();

	for (const auto & points : processes) {
		if (points.size () > 0) {
			min = std::min (min, points[0]);
			max = std::max (max, points[points.size () - 1]);
		}
	}

	if (min <= max) {
		return max - min;
	} else {
		return 0; // If there are no points at all, return 0
	}
}

/* A sliding cursor is an iterator over the coordinates of points, all shifted by the given shift.
 * This is a tool used in some computations below.
 */
struct SlidingCursor {
	const SortedVec<Point> & points; // Points to slide on
	const PointSpace shift;          // Shifting from points

	static constexpr PointSpace inf = std::numeric_limits<PointSpace>::max ();

	// Indexes of next point to be visited, and shifted value (or inf)
	size_t current_i = 0;
	Point current_x;

	SlidingCursor (const SortedVec<Point> & points, PointSpace shift) : points (points), shift (shift) {
		current_x = get_shifted_point (0);
	}
	Point get_shifted_point (size_t i) const {
		if (i < points.size ()) {
			return points[i] + shift;
		} else {
			return inf;
		}
	}
	void advance_if_equal (Point new_x) {
		if (new_x == current_x) {
			current_i += 1;
			current_x = get_shifted_point (current_i);
		}
	}
};

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

	size_t starting_i_m = 0;
	for (const Point x_l : l_points) {
		// x_l = N_l[i_l], with N_l[x] a strictly increasing function of x.
		// Compute shape(x_m - x_l) for all x_m in (x_l + non_zero_domain) interval.
		const auto interval_i_l = x_l + non_zero_domain;

		// starting_i_m = min{i_m, N_m[i_m] - N_l[i_l] >= non_zero_domain.left}.
		// We can restrict the search by starting from:
		// last_starting_i_m = min{i_m, N_m[i_m] - N_l[i_l - 1] >= non_zero_domain.left or i_m == 0}.
		// We have: N_m[starting_i_m] >= N_l[i_l] + nzd.left > N_l[i_l - 1] + nzd.left.
		// Because N_m is increasing and properties of the min, starting_i_m >= last_starting_i_m.
		while (starting_i_m < m_points.size () && !(interval_i_l.left <= m_points[starting_i_m])) {
			starting_i_m += 1;
		}
		if (starting_i_m == m_points.size ()) {
			// starting_i_m is undefined because last(N_m) < N_l[i_l] + non_zero_domain.left.
			// last(N_m) == max(x_m in N_m) because N_m[x] is strictly increasing.
			// So for each j > i_l , max(x_m) < N[j] + non_zero_domain.left, and shape (x_m - N_l[j]) == 0.
			// We can stop there as the sum is already complete.
			break;
		}
		// Sum values of shape(x_m - x_l) as long as x_m is in interval_i_l.
		// starting_i_m defined => for each i_m < starting_i_m, shape(N_m[i_m] - x_l) == 0.
		// Thus we only scan from starting_i_m to the last i_m in interval.
		// N_m[x] is strictly increasing so we only need to check the right bound of the interval.
		for (size_t i_m = starting_i_m; i_m < m_points.size () && m_points[i_m] <= interval_i_l.right; i_m += 1) {
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

/* Compute sup_{x} sum_{x_l in N_l} interval(x - x_l).
 * This is a building block for computation of B_hat, used in the computation of lasso penalties (d).
 *
 * For an indicator interval, the sum is a piecewise constant function of x.
 * This function has at maximum 2*|N_l| points of change, so the number of different values is finite.
 * Thus the sup over x is a max over all these possible values.
 */
inline int32_t sup_of_sum_of_differences_to_points (const SortedVec<Point> & points,
                                                    shape::IntervalIndicator indicator) {
	int32_t max = std::numeric_limits<int32_t>::min ();

	// These structs represent the sets of left and right interval bounds coordinates.
	SlidingCursor left_interval_bounds (points, -indicator.half_width);
	SlidingCursor right_interval_bounds (points, indicator.half_width);

	while (true) {
		// Loop over all interval boundaries: x is a {left, right, both} interval bound.
		const Point x = std::min (left_interval_bounds.current_x, right_interval_bounds.current_x);
		if (x == SlidingCursor::inf) {
			break; // No more points to process.
		}
		// The sum of intervals at x is the number of entered intervals minus the number of exited intervals.
		// Thus the sum is the difference between the indexes of the left bound iterator and the right one.
		// Because the interval is a closed one, we advance the entering bound before computing the sum.
		left_interval_bounds.advance_if_equal (x);
		assert (left_interval_bounds.current_i >= right_interval_bounds.current_i);
		const auto sum_value_for_x = PointSpace (left_interval_bounds.current_i - right_interval_bounds.current_i);
		max = std::max (max, sum_value_for_x);
		right_interval_bounds.advance_if_equal (x);
	}
	return max;
}

// Scaling can be moved out
template <typename T, typename Inner>
inline auto sup_of_sum_of_differences_to_points (const SortedVec<Point> & points,
                                                 const shape::Scaled<T, Inner> & shape) {
	return shape.scale * sup_of_sum_of_differences_to_points (points, shape.inner);
}
// Shifting has no effect on the sup value.
template <typename Inner>
inline auto sup_of_sum_of_differences_to_points (const SortedVec<Point> & points, const shape::Shifted<Inner> & shape) {
	return sup_of_sum_of_differences_to_points (points, shape.inner);
}

// Conversion of objects to shapes (shape.h)
inline auto to_shape (HistogramBase::Interval i) {
	// TODO Histo::Interval is ]left; right], but shape::Interval is [left; right].
	const auto delta = i.right - i.left;
	const auto center = (i.left + i.right) / 2;
	return shape::scaled (1. / std::sqrt (delta), shape::shifted (center, shape::IntervalIndicator::with_width (delta)));
}
inline auto to_shape (IntervalKernel kernel) {
	return shape::scaled (1. / std::sqrt (kernel.width), shape::IntervalIndicator::with_width (kernel.width));
}

/******************************************************************************
 * Penalty values for lassoshooting.
 */
inline Matrix_M_MK1 compute_d (double gamma, span<const Matrix_M_MK1> b_by_region, const Matrix_M_MK1 & b_hat) {
	const auto nb_regions = b_by_region.size ();
	assert (nb_regions > 0);
	const auto nb_processes = b_by_region[0].nb_processes;
	const auto base_size = b_by_region[0].base_size;

	const auto log_factor = std::log (nb_processes + nb_processes * nb_processes * base_size);

	// Compute V_hat_m_kl * R^2. Division by R^2 is done later.
	Matrix_M_MK1 v_hat (nb_processes, base_size);
	for (const auto & b : b_by_region) {
		v_hat.m_lk_values ().array () += b.m_lk_values ().array ().square ();
	}
	const auto v_hat_factor = (1. / double(nb_regions * nb_regions)) * 2. * gamma * log_factor;

	// B_hat_mkl is computed externally
	const auto b_hat_factor = gamma * log_factor / 3.;

	Matrix_M_MK1 d (nb_processes, base_size);
	d.m_0_values ().setZero (); // No penalty for constant component of estimators
	d.m_lk_values () =
	    (v_hat_factor * v_hat.m_lk_values ().array ()).sqrt () + b_hat_factor * b_hat.m_lk_values ().array ();
	return d;
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
	std::vector<size_t> sliding_interval_bounds (base.base_size + 1, 0);

	const auto n_m = m_points.size ();
	for (const Point x_l : l_points) {
		// Compute indexes of all k interval boundaries, shifted from the current x_l.
		// This can be done by searching m points starting at the previous positions (x_l increased).
		for (size_t k = 0; k < base.base_size + 1; ++k) {
			const PointSpace shift = PointSpace (k) * base.delta;
			size_t i = sliding_interval_bounds[k];
			while (i < n_m && !(m_points[i] - x_l > shift)) {
				i += 1;
			}
			sliding_interval_bounds[k] = i;
		}
		// Accumulate the number of points in each shifted interval for the current x_l.
		// Number of points = difference between indexes of interval boundaries.
		for (size_t k = 0; k < base.base_size; ++k) {
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
 * The strategy is to compute the integral by splitting the point space in the constant parts of N_l(..) * N_l2(..).
 * Thus we loop over all points of changes of this product.
 */
inline int64_t compute_g_ll2kk2_histogram_integral (const SortedVec<Point> & l_points,
                                                    const SortedVec<Point> & l2_points,
                                                    HistogramBase::Interval interval,
                                                    HistogramBase::Interval interval2) {
	struct SlidingInterval {
		// Iterators over coordinates where points enter of exit the sliding interval
		SlidingCursor entering;
		SlidingCursor exiting;

		/* For a point p, and an interval ]x - (k+1)*delta, x - k*delta], moving the interval from left to right:
		 * p enters when x - k*delta = p <=> x = p+k*delta <=> x = p + interval.left.
		 * p exits when x - (k+1)*delta = p <=> x = p + (k+1)*delta <=> x = p + interval.right.
		 * Thus the coordinate iterators are points shifted by i.left / i.right.
		 */
		SlidingInterval (const SortedVec<Point> & points, HistogramBase::Interval i)
		    : entering (points, i.left), exiting (points, i.right) {}

		Point min_interesting_x () const { return std::min (entering.current_x, exiting.current_x); }
		size_t current_points_inside () const {
			assert (entering.current_i >= exiting.current_i);
			return entering.current_i - exiting.current_i;
		}
		void point_processed (Point x) {
			entering.advance_if_equal (x);
			exiting.advance_if_equal (x);
		}
	};
	SlidingInterval si1 (l_points, interval);
	SlidingInterval si2 (l2_points, interval2);

	int64_t accumulated_area = 0;
	Point previous_x = std::numeric_limits<Point>::min (); // Start at -inf

	while (true) {
		// Process the next interesting coordinate
		const Point x = std::min (si1.min_interesting_x (), si2.min_interesting_x ());
		if (x == SlidingCursor::inf) {
			// No more points to process, all next points are inf.
			assert (si1.current_points_inside () == 0);
			assert (si2.current_points_inside () == 0);
			break;
		}
		// Integrate the constant between current and previous x.
		accumulated_area += int64_t (si1.current_points_inside () * si2.current_points_inside ()) * (x - previous_x);
		// Point x has been processed, move to next one
		previous_x = x;
		si1.point_processed (x);
		si2.point_processed (x);
	}
	return accumulated_area;
}

// Complexity: O( M^2 * K * max(|N_m|) ).
inline Matrix_M_MK1 compute_b (span<const SortedVec<Point>> processes, HistogramBase base) {
	const auto nb_processes = processes.size ();
	const auto base_size = base.base_size;
	const auto inv_sqrt_delta = 1. / std::sqrt (double(base.delta));
	Matrix_M_MK1 b (nb_processes, base_size);

	for (ProcessId m = 0; m < nb_processes; ++m) {
		const auto & m_process = processes[m];

		// b_0
		b.set_0 (m, double(m_process.size ()));

		// b_lk
		for (ProcessId l = 0; l < nb_processes; ++l) {
			auto counts = compute_b_ml_histogram_counts_for_all_k (m_process, processes[l], base);
			for (FunctionBaseId k = 0; k < base_size; ++k) {
				b.set_lk (m, l, k, double(counts[k]) * inv_sqrt_delta);
			}
		}
	}
	return b;
}

// Complexity: O( M^2 * K * max(|N_m|) ).
inline MatrixG compute_g (span<const SortedVec<Point>> processes, HistogramBase base) {
	const auto nb_processes = processes.size ();
	const auto base_size = base.base_size;
	const auto sqrt_delta = std::sqrt (double(base.delta));
	const auto inv_delta = 1 / double(base.delta);
	MatrixG g (nb_processes, base_size);

	g.set_tmax (tmax (processes));

	for (ProcessId l = 0; l < nb_processes; ++l) {
		const auto g_lk = double(processes[l].size ()) * sqrt_delta;
		for (FunctionBaseId k = 0; k < base_size; ++k) {
			g.set_g (l, k, g_lk);
		}
	}

	auto G_value = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
		const auto integral =
		    compute_g_ll2kk2_histogram_integral (processes[l], processes[l2], base.interval (k), base.interval (k2));
		return double(integral) * inv_delta;
	};
	/* G symmetric, only compute for (l2,k2) >= (l,k) (lexicographically).
	 *
	 * G_{l,l2,k,k2} = integral_x sum_{x_l,x_l2} phi_k (x - x_l) phi_k2 (x - x_l2) dx.
	 * By change of variable x -> x + c*delta: G_{l,l2,k,k2} = G_{l,l2,k+c,k2+c}.
	 * Thus for each block of G for (l,l2), we only need to compute 2K+1 values of G.
	 * We compute the values on two borders, and copy the values for all inner cells.
	 */
	for (ProcessId l = 0; l < nb_processes; ++l) {
		// Case l2 == l: compute for k2 >= k:
		for (FunctionBaseId k = 0; k < base_size; ++k) {
			// Compute G_{l,l,0,k} and copy to G_{l,l,c,k+c} for k in [0,K[.
			const auto v = G_value (l, l, FunctionBaseId{0}, k);
			for (size_t c = 0; k + c < base_size; ++c) {
				g.set_G (l, l, FunctionBaseId{c}, FunctionBaseId{k + c}, v);
			}
		}
		// Case l2 > l: compute for all (k,k2):
		for (ProcessId l2 = l; l2 < nb_processes; ++l2) {
			// Compute G_{l,l2,0,0} and copy to G_{l,l2,c,c}.
			{
				const auto v = G_value (l, l2, FunctionBaseId{0}, FunctionBaseId{0});
				for (size_t c = 0; c < base_size; ++c) {
					g.set_G (l, l2, FunctionBaseId{c}, FunctionBaseId{c}, v);
				}
			}
			/* for k in [1,K[:
			 * Compute G_{l,l2,0,k} and copy to G_{l,l2,c,k+c}.
			 * Compute G_{l,l2,k,0} and copy to G_{l,l2,k+c,c}.
			 */
			for (FunctionBaseId k = 1; k < base_size; ++k) {
				const auto v_0k = G_value (l, l2, FunctionBaseId{0}, k);
				const auto v_k0 = G_value (l, l2, k, FunctionBaseId{0});
				for (size_t c = 0; k + c < base_size; ++c) {
					g.set_G (l, l2, FunctionBaseId{c}, FunctionBaseId{k + c}, v_0k);
					g.set_G (l, l2, FunctionBaseId{k + c}, FunctionBaseId{c}, v_k0);
				}
			}
		}
	}
	return g;
}

inline Matrix_M_MK1 compute_b_hat (const ProcessesRegionData & processes, HistogramBase base) {
	// TODO
}

/******************************************************************************
 * Histogram with interval convolution kernels.
 */
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
                                                   const SortedVec<Point> & l2_points, PointSpace delta,
                                                   FunctionBaseId k, FunctionBaseId k2, IntervalKernel kernel,
                                                   IntervalKernel kernel2) {
	// V = sum_{x_l,x_l2} corr(conv(W_l,phi_k),conv(W_l2,phi_k2)) (x_l-x_l2)
	// V = factorized_scaling * sum_{x_l,x_l2} shifted((k2-k)*delta, conv(trapezoid(l), trapezoid(l2))) (x_l-x_l2)

	// prepare common scaling and shift
	const auto factorized_scaling = 1. / (double(delta) * std::sqrt (kernel.width * kernel2.width));
	const auto factorized_shift = delta * (PointSpace (k2) - PointSpace (k));
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
	const auto nb_processes = processes.size ();
	const auto base_size = base.base_size;
	Matrix_M_MK1 b (nb_processes, base_size);

	for (ProcessId m = 0; m < nb_processes; ++m) {
		const auto & m_process = processes[m];
		const auto & m_kernel = kernels[m];

		// b0
		b.set_0 (m, double(m_process.size ()) * std::sqrt (m_kernel.width));

		// b_lk
		for (ProcessId l = 0; l < nb_processes; ++l) {
			const auto & l_kernel = kernels[l];
			for (FunctionBaseId k = 0; k < base_size; ++k) {
				const auto v = compute_b_mlk_histogram (m_process, processes[l], base.interval (k), m_kernel, l_kernel);
				b.set_lk (m, l, k, v);
			}
		}
	}
	return b;
}

inline MatrixG compute_g (span<const SortedVec<Point>> processes, HistogramBase base,
                          span<const IntervalKernel> kernels) {
	assert (kernels.size () == processes.size ());
	const auto nb_processes = processes.size ();
	const auto base_size = base.base_size;
	const auto sqrt_delta = std::sqrt (double(base.delta));
	MatrixG g (nb_processes, base_size);

	g.set_tmax (tmax (processes));

	for (ProcessId l = 0; l < nb_processes; ++l) {
		/* g_lk = sum_{x_m} integral convolution(w_l,phi_k) (x - x_m) dx.
		 * g_lk = sum_{x_m} (integral w_l) (integral phi_k) = sum_{x_m} eta_l sqrt(delta) = |N_m| eta_l sqrt(delta).
		 */
		const auto g_lk = double(processes[l].size ()) * std::sqrt (kernels[l].width) * sqrt_delta;
		for (FunctionBaseId k = 0; k < base_size; ++k) {
			g.set_g (l, k, g_lk);
		}
	}

	auto G_value = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
		return compute_g_ll2kk2_histogram_integral (processes[l], processes[l2], base.delta, k, k2, kernels[l],
		                                            kernels[l2]);
	};
	/* G symmetric, only compute for (l2,k2) >= (l,k) (lexicographically).
	 *
	 * G_{l,l2,k,k2} = integral_x sum_{x_l,x_l2} conv(w_l,phi_k) (x - x_l) conv(w_l2,phi_k2) (x - x_l2) dx.
	 * By change of variable x -> x + c*delta: G_{l,l2,k,k2} = G_{l,l2,k+c,k2+c}.
	 * Thus for each block of G for (l,l2), we only need to compute 2K+1 values of G.
	 * We compute the values on two borders, and copy the values for all inner cells.
	 */
	for (ProcessId l = 0; l < nb_processes; ++l) {
		// Case l2 == l: compute for k2 >= k:
		for (FunctionBaseId k = 0; k < base_size; ++k) {
			// Compute G_{l,l,0,k} and copy to G_{l,l,c,k+c} for k in [0,K[.
			const auto v = G_value (l, l, FunctionBaseId{0}, k);
			for (size_t c = 0; k + c < base_size; ++c) {
				g.set_G (l, l, FunctionBaseId{c}, FunctionBaseId{k + c}, v);
			}
		}
		// Case l2 > l: compute for all (k,k2):
		for (ProcessId l2 = l; l2 < nb_processes; ++l2) {
			// Compute G_{l,l2,0,0} and copy to G_{l,l2,c,c}.
			{
				const auto v = G_value (l, l2, FunctionBaseId{0}, FunctionBaseId{0});
				for (size_t c = 0; c < base_size; ++c) {
					g.set_G (l, l2, FunctionBaseId{c}, FunctionBaseId{c}, v);
				}
			}
			/* for k in [1,K[:
			 * Compute G_{l,l2,0,k} and copy to G_{l,l2,c,k+c}.
			 * Compute G_{l,l2,k,0} and copy to G_{l,l2,k+c,c}.
			 */
			for (FunctionBaseId k = 1; k < base_size; ++k) {
				const auto v_0k = G_value (l, l2, FunctionBaseId{0}, k);
				const auto v_k0 = G_value (l, l2, k, FunctionBaseId{0});
				for (size_t c = 0; k + c < base_size; ++c) {
					g.set_G (l, l2, FunctionBaseId{c}, FunctionBaseId{k + c}, v_0k);
					g.set_G (l, l2, FunctionBaseId{k + c}, FunctionBaseId{c}, v_k0);
				}
			}
		}
	}
	return g;
}
