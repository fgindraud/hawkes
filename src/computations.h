#pragma once

#include <cmath>
#include <limits>

#include "lassoshooting.h"
#include "shape.h"
#include "types.h"

/******************************************************************************
 * Generic building blocks useful for all cases.
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
		return 0.; // If there are no points at all, return 0
	}
}

/* A sliding cursor is an iterator over the coordinates of points, all shifted by the given shift.
 * This is a tool used in some computations below.
 */
struct SlidingCursor {
	const SortedVec<Point> & points; // Points to slide on
	const PointSpace shift;          // Shifting from points

	static constexpr PointSpace inf = std::numeric_limits<PointSpace>::infinity ();

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
inline double sum_of_point_differences (const SortedVec<Point> & m_points, const SortedVec<Point> & l_points,
                                        const Shape & shape) {
	double sum = 0.;

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
			sum += shape (m_points[i_m] - x_l);
		}
	}

	return sum;
}

// Scaling can be moved out of computation.
template <typename Inner>
inline double sum_of_point_differences (const SortedVec<Point> & m_points, const SortedVec<Point> & l_points,
                                        const shape::Scaled<Inner> & shape) {
	return shape.scale * sum_of_point_differences (m_points, l_points, shape.inner);
}

/* Compute sup_{x} sum_{x_l in N_l} interval(x - x_l).
 * This is a building block for computation of B_hat, used in the computation of lasso penalties (d).
 *
 * For an indicator interval, the sum is a piecewise constant function of x.
 * This function has at maximum 2*|N_l| points of change, so the number of different values is finite.
 * Thus the sup over x is a max over all these possible values.
 */
inline double sup_of_sum_of_differences_to_points (const SortedVec<Point> & points,
                                                   shape::IntervalIndicator indicator) {
	// These structs represent the sets of left and right interval bounds coordinates.
	SlidingCursor left_interval_bounds (points, -indicator.half_width);
	SlidingCursor right_interval_bounds (points, indicator.half_width);
	double max = -std::numeric_limits<double>::infinity ();
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
inline double sup_of_sum_of_differences_to_points (const SortedVec<Point> & points, HistogramBase::Interval interval) {
	// Same algorithm, but left bound is advanced after computing the sum due to the open left bound.
	SlidingCursor left_interval_bounds (points, interval.left);
	SlidingCursor right_interval_bounds (points, interval.right);
	double max = -std::numeric_limits<double>::infinity ();
	while (true) {
		const Point x = std::min (left_interval_bounds.current_x, right_interval_bounds.current_x);
		if (x == SlidingCursor::inf) {
			break;
		}
		assert (left_interval_bounds.current_i >= right_interval_bounds.current_i);
		const auto sum_value_for_x = PointSpace (left_interval_bounds.current_i - right_interval_bounds.current_i);
		max = std::max (max, sum_value_for_x);
		left_interval_bounds.advance_if_equal (x);
		right_interval_bounds.advance_if_equal (x);
	}
	return max;
}

// Scaling can be moved out
template <typename Inner>
inline double sup_of_sum_of_differences_to_points (const SortedVec<Point> & points,
                                                   const shape::Scaled<Inner> & shape) {
	return shape.scale * sup_of_sum_of_differences_to_points (points, shape.inner);
}
// Shifting has no effect on the sup value.
template <typename Inner>
inline double sup_of_sum_of_differences_to_points (const SortedVec<Point> & points,
                                                   const shape::Shifted<Inner> & shape) {
	return sup_of_sum_of_differences_to_points (points, shape.inner);
}

// Conversion of objects to shapes (shape.h)
inline auto to_shape (HistogramBase::Interval i) {
	// Histo::Interval is ]left; right], but shape::Interval is [left; right].
	// This conversion is only valid if used in a convolution, where the type of interval bound does not matter !
	const auto delta = i.right - i.left;
	const auto center = (i.left + i.right) / 2.;
	return shape::scaled (1. / std::sqrt (delta), shape::shifted (center, shape::IntervalIndicator::with_width (delta)));
}
inline auto to_shape (IntervalKernel kernel) {
	return shape::scaled (normalization_factor (kernel), shape::IntervalIndicator::with_width (kernel.width));
}

/******************************************************************************
 * Basic histogram case.
 * Due to the simplicity of the functions involved, the computations can use efficient specialized algorithms.
 */

/* Compute b_{m,l,k} * sqrt(delta) for all k, for the histogram base.
 * Return a vector with the k values.
 * Complexity is O( |N_l| + (K+1) * |N_m| ).
 *
 * In the histogram case, b_{m,l,k} = sum_{(x_l,x_m) in (N_l,N_m), k*delta < x_m - x_l <= (k+1)*delta} 1/sqrt(delta).
 * The strategy is to count points of N_l in the interval ] k*delta + x_l, (k+1)*delta + x_l ] for each x_l.
 * This specific functions does it for all k at once.
 * This is more efficient because the upper bound of the k-th interval is the lower bound of the (k+1)-th.
 * Thus we compute the bounds only once.
 */
inline std::vector<int64_t> b_ml_histogram_counts_for_all_k_denormalized (const SortedVec<Point> & m_points,
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
inline double g_ll2kk2_histogram_integral_denormalized (const SortedVec<Point> & l_points,
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

	double accumulated_area = 0;
	Point previous_x = -std::numeric_limits<Point>::max (); // Start at -inf
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
		accumulated_area += double(si1.current_points_inside () * si2.current_points_inside ()) * (x - previous_x);
		// Point x has been processed, move to next one
		previous_x = x;
		si1.point_processed (x);
		si2.point_processed (x);
	}
	return accumulated_area;
}

// Complexity: O( M^2 * K * max(|N_m|) ).
inline Matrix_M_MK1 compute_b (span<const SortedVec<Point>> points, HistogramBase base, None /*kernels*/) {
	const auto nb_processes = points.size ();
	const auto base_size = base.base_size;
	const auto phi_normalization_factor = normalization_factor (base);
	Matrix_M_MK1 b (nb_processes, base_size);

	for (ProcessId m = 0; m < nb_processes; ++m) {
		// b_0
		b.set_0 (m, double(points[m].size ()));
		// b_lk
		for (ProcessId l = 0; l < nb_processes; ++l) {
			const auto counts = b_ml_histogram_counts_for_all_k_denormalized (points[m], points[l], base);
			for (FunctionBaseId k = 0; k < base_size; ++k) {
				b.set_lk (m, l, k, double(counts[k]) * phi_normalization_factor);
			}
		}
	}
	return b;
}

// Complexity: O( M^2 * K * max(|N_m|) ).
inline MatrixG compute_g (span<const SortedVec<Point>> points, HistogramBase base, None /*kernels*/) {
	const auto nb_processes = points.size ();
	const auto base_size = base.base_size;
	const auto sqrt_delta = std::sqrt (base.delta);
	const auto inv_delta = 1. / base.delta;
	MatrixG g (nb_processes, base_size);

	g.set_tmax (tmax (points));

	for (ProcessId l = 0; l < nb_processes; ++l) {
		const auto g_lk = double(points[l].size ()) * sqrt_delta;
		for (FunctionBaseId k = 0; k < base_size; ++k) {
			g.set_g (l, k, g_lk);
		}
	}

	auto G_value = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
		const auto integral =
		    g_ll2kk2_histogram_integral_denormalized (points[l], points[l2], base.interval (k), base.interval (k2));
		return integral * inv_delta;
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

// Complexity: O( M^2 * K + M * max(|N_m|) )
// Computation is exact (sup can be evaluated perfectly).
inline Matrix_M_MK1 compute_b_hat (const DataByProcessRegion<SortedVec<Point>> & points, HistogramBase base,
                                   None /*kernels*/) {
	const auto nb_processes = points.nb_processes ();
	const auto nb_regions = points.nb_regions ();
	const auto base_size = base.base_size;
	Matrix_M_MK1 b_hat (nb_processes, base_size);

	// B_hat_{m,l,k} = sum_region sup_x sum_{x_l in N_l_region} phi_k (x - x_l)
	// sup_x sum_{x_l in N_l_region} phi_k (x - x_l) is not affected by shifting, thus does not depend on k.
	// Thus B_hat_{m,l,k} = B_hat_l, as the expression is independent of m and k.
	const auto phi_normalization_factor = normalization_factor (base);
	const auto phi_0_interval = base.interval (FunctionBaseId{0}); // Representative for all k.

	for (ProcessId l = 0; l < nb_processes; ++l) {
		double sum_of_region_sups = 0.;
		for (RegionId r = 0; r < nb_regions; ++r) {
			sum_of_region_sups += sup_of_sum_of_differences_to_points (points.data (l, r), phi_0_interval);
		}
		const auto b_hat_l = sum_of_region_sups * phi_normalization_factor;

		for (ProcessId m = 0; m < nb_processes; ++m) {
			for (FunctionBaseId k = 0; k < base_size; ++k) {
				b_hat.set_lk (m, l, k, b_hat_l);
			}
		}
	}
	return b_hat;
}

/******************************************************************************
 * Histogram with interval convolution kernels.
 *
 * In this case we use the shape strategy.
 * Values of B/G are expressed from convolution of shapes coming from the base and kernels.
 * The convoluted shapes are computed using the template-expression strategy in shape.h.
 * Then the B and G values are computed using the shape expressions.
 * This is less efficient than the Histogram/no_kernel case.
 *
 * TODO doc.
 */
inline Matrix_M_MK1 compute_b (span<const SortedVec<Point>> points, HistogramBase base,
                               const std::vector<IntervalKernel> & kernels) {
	assert (kernels.size () == points.size ());
	const auto nb_processes = points.size ();
	const auto base_size = base.base_size;
	Matrix_M_MK1 b (nb_processes, base_size);

	const auto b_mlk = [](const SortedVec<Point> & m_points, const SortedVec<Point> & l_points,
	                      HistogramBase::Interval base_interval, IntervalKernel m_kernel, IntervalKernel l_kernel) {
		const auto shape = convolution (to_shape (base_interval), convolution (to_shape (m_kernel), to_shape (l_kernel)));
		return sum_of_point_differences (m_points, l_points, shape);
	};
	for (ProcessId m = 0; m < nb_processes; ++m) {
		// b0
		b.set_0 (m, double(points[m].size ()) * std::sqrt (kernels[m].width));
		// b_lk
		for (ProcessId l = 0; l < nb_processes; ++l) {
			for (FunctionBaseId k = 0; k < base_size; ++k) {
				const auto v = b_mlk (points[m], points[l], base.interval (k), kernels[m], kernels[l]);
				b.set_lk (m, l, k, v);
			}
		}
	}
	return b;
}

inline MatrixG compute_g (span<const SortedVec<Point>> points, HistogramBase base,
                          const std::vector<IntervalKernel> & kernels) {
	assert (kernels.size () == points.size ());
	const auto nb_processes = points.size ();
	const auto base_size = base.base_size;
	const auto sqrt_delta = std::sqrt (double(base.delta));
	MatrixG g (nb_processes, base_size);

	g.set_tmax (tmax (points));

	for (ProcessId l = 0; l < nb_processes; ++l) {
		/* g_lk = sum_{x_m} integral convolution(w_l,phi_k) (x - x_m) dx.
		 * g_lk = sum_{x_m} (integral w_l) (integral phi_k) = sum_{x_m} eta_l sqrt(delta) = |N_m| eta_l sqrt(delta).
		 */
		const auto g_lk = double(points[l].size ()) * std::sqrt (kernels[l].width) * sqrt_delta;
		for (FunctionBaseId k = 0; k < base_size; ++k) {
			g.set_g (l, k, g_lk);
		}
	}

	const auto G_ll2kk2 = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
		// V = sum_{x_l,x_l2} corr(conv(W_l,phi_k),conv(W_l2,phi_k2)) (x_l-x_l2)
		// V = factorized_scaling * sum_{x_l,x_l2} shifted((k2-k)*delta, conv(trapezoid(l), trapezoid(l2))) (x_l-x_l2)

		// compute the shape
		const auto phi_indicator = shape::IntervalIndicator::with_width (base.delta);
		const auto kernel_indicator = shape::IntervalIndicator::with_width (kernels[l].width);
		const auto kernel2_indicator = shape::IntervalIndicator::with_width (kernels[l2].width);
		const auto trapezoid = convolution (kernel_indicator, phi_indicator);
		const auto trapezoid2 = convolution (kernel2_indicator, phi_indicator);
		const auto shape = convolution (trapezoid, trapezoid2);
		// Apply final transformations and compute sum
		const auto factorized_scaling = 1. / (base.delta * std::sqrt (kernels[l].width * kernels[l2].width));
		const auto factorized_shift = base.delta * (PointSpace (k2) - PointSpace (k));
		const auto final_shape = scaled (factorized_scaling, shifted (factorized_shift, shape));
		return sum_of_point_differences (points[l], points[l2], final_shape);
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
			const auto v = G_ll2kk2 (l, l, FunctionBaseId{0}, k);
			for (size_t c = 0; k + c < base_size; ++c) {
				g.set_G (l, l, FunctionBaseId{c}, FunctionBaseId{k + c}, v);
			}
		}
		// Case l2 > l: compute for all (k,k2):
		for (ProcessId l2 = l; l2 < nb_processes; ++l2) {
			// Compute G_{l,l2,0,0} and copy to G_{l,l2,c,c}.
			{
				const auto v = G_ll2kk2 (l, l2, FunctionBaseId{0}, FunctionBaseId{0});
				for (size_t c = 0; c < base_size; ++c) {
					g.set_G (l, l2, FunctionBaseId{c}, FunctionBaseId{c}, v);
				}
			}
			/* for k in [1,K[:
			 * Compute G_{l,l2,0,k} and copy to G_{l,l2,c,k+c}.
			 * Compute G_{l,l2,k,0} and copy to G_{l,l2,k+c,c}.
			 */
			for (FunctionBaseId k = 1; k < base_size; ++k) {
				const auto v_0k = G_ll2kk2 (l, l2, FunctionBaseId{0}, k);
				const auto v_k0 = G_ll2kk2 (l, l2, k, FunctionBaseId{0});
				for (size_t c = 0; k + c < base_size; ++c) {
					g.set_G (l, l2, FunctionBaseId{c}, FunctionBaseId{k + c}, v_0k);
					g.set_G (l, l2, FunctionBaseId{k + c}, FunctionBaseId{c}, v_k0);
				}
			}
		}
	}
	return g;
}

inline Matrix_M_MK1 compute_b_hat (const DataByProcessRegion<SortedVec<Point>> & points, HistogramBase base,
                                   const std::vector<IntervalKernel> & kernels) {
	assert (kernels.size () == points.nb_processes ());
	const auto nb_processes = points.nb_processes ();
	const auto nb_regions = points.nb_regions ();
	const auto base_size = base.base_size;
	Matrix_M_MK1 b_hat (nb_processes, base_size);

	for (ProcessId l = 0; l < nb_processes; ++l) {
		for (ProcessId m = 0; m < nb_processes; ++m) {
			for (FunctionBaseId k = 0; k < base_size; ++k) {
				// Base shapes
				const auto w_m = to_shape (kernels[m]);
				const auto w_l = to_shape (kernels[l]);
				const auto phi_k = to_shape (base.interval (k));
				// Approximate trapezoids with an interval of height=max, width=width of trapezoid
				const auto intermediate = interval_approximation (convolution (w_l, phi_k));
				const auto approx = interval_approximation (convolution (intermediate, w_m));

				double sum_of_region_sups = 0;
				for (RegionId r = 0; r < nb_regions; ++r) {
					sum_of_region_sups += sup_of_sum_of_differences_to_points (points.data (l, r), approx);
				}
				b_hat.set_lk (m, l, k, sum_of_region_sups);
			}
		}
	}
	return b_hat;
}

/******************************************************************************
 * FIXME experimental: point specific kernels
 * Data set is set of {point, kernel-width} instead of just points.
 * Data sets are not sorted.
 */
inline Matrix_M_MK1 compute_b (span<const std::vector<PointAndKernel>> processes, HistogramBase base) {
	const auto nb_processes = processes.size ();
	const auto base_size = base.base_size;
	Matrix_M_MK1 b (nb_processes, base_size);

	const auto sum_sqrt_kernel_width = [](const std::vector<PointAndKernel> & d) {
		double sum = 0.;
		for (const auto & t : d) {
			sum += std::sqrt (t.kernel.width);
		}
		return sum;
	};
	const auto b_mlk = [](const std::vector<PointAndKernel> & m, const std::vector<PointAndKernel> & l,
	                      HistogramBase::Interval phi_k) {
		double sum = 0.;
		const auto phi_shape = to_shape (phi_k);
		for (const auto & pm : m) {
			for (const auto & pl : l) {
				const auto shape = convolution (phi_shape, convolution (to_shape (pm.kernel), to_shape (pl.kernel)));
				sum += shape (pm.point - pl.point);
			}
		}
		return sum;
	};

	for (ProcessId m = 0; m < nb_processes; ++m) {
		// b0
		b.set_0 (m, sum_sqrt_kernel_width (processes[m]));
		// b_lk
		for (ProcessId l = 0; l < nb_processes; ++l) {
			for (FunctionBaseId k = 0; k < base_size; ++k) {
				b.set_lk (m, l, k, b_mlk (processes[m], processes[l], base.interval (k)));
			}
		}
	}
	return b;
}
inline MatrixG compute_g (span<const std::vector<PointAndKernel>> processes, HistogramBase base) {
	const auto nb_processes = processes.size ();
	const auto base_size = base.base_size;
	const auto sqrt_delta = std::sqrt (base.delta);
	MatrixG g (nb_processes, base_size);

	const auto tmax = [](span<const std::vector<PointAndKernel>> processes) {
		PointSpace min = std::numeric_limits<PointSpace>::max ();
		PointSpace max = std::numeric_limits<PointSpace>::min ();
		for (const auto & points : processes) {
			const auto local_minmax = std::minmax_element (
			    points.begin (), points.end (), [](const auto & lhs, const auto & rhs) { return lhs.point < rhs.point; });

			if (local_minmax.first != points.end ()) {
				min = std::min (min, local_minmax.first->point);
			}
			if (local_minmax.second != points.end ()) {
				max = std::max (max, local_minmax.second->point);
			}
		}
		if (min <= max) {
			return max - min;
		} else {
			return 0.; // If there are no points at all, return 0
		}
	};
	g.set_tmax (tmax (processes));

	const auto sum_sqrt_kernel_width = [](const std::vector<PointAndKernel> & d) {
		double sum = 0.;
		for (const auto & t : d) {
			sum += std::sqrt (t.kernel.width);
		}
		return sum;
	};
	for (ProcessId l = 0; l < nb_processes; ++l) {
		/* g_lk = sum_{x_m} integral convolution(w_l,phi_k) (x - x_m) dx.
		 * g_lk = sum_{x_m} (integral w_l) (integral phi_k) = sum_{x_m} eta_l sqrt(delta) = |N_m| eta_l sqrt(delta).
		 */
		const auto g_lk = sum_sqrt_kernel_width (processes[l]) * sqrt_delta;
		for (FunctionBaseId k = 0; k < base_size; ++k) {
			g.set_g (l, k, g_lk);
		}
	}

	const auto G_ll2kk2 = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
		double sum = 0.;
		for (const auto & lp : processes[l]) {
			for (const auto & l2p : processes[l2]) {
				// compute the shape
				const auto phi_indicator = shape::IntervalIndicator::with_width (base.delta);
				const auto kernel_indicator = shape::IntervalIndicator::with_width (lp.kernel.width);
				const auto kernel2_indicator = shape::IntervalIndicator::with_width (l2p.kernel.width);
				const auto trapezoid = convolution (kernel_indicator, phi_indicator);
				const auto trapezoid2 = convolution (kernel2_indicator, phi_indicator);
				const auto shape = convolution (trapezoid, trapezoid2);
				// Apply final transformations and compute sum
				const auto factorized_scaling = 1. / (base.delta * std::sqrt (lp.kernel.width * l2p.kernel.width));
				const auto factorized_shift = base.delta * (PointSpace (k2) - PointSpace (k));
				const auto final_shape = scaled (factorized_scaling, shifted (factorized_shift, shape));
				sum += final_shape (lp.point - l2p.point);
			}
		}
		return sum;
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
			const auto v = G_ll2kk2 (l, l, FunctionBaseId{0}, k);
			for (size_t c = 0; k + c < base_size; ++c) {
				g.set_G (l, l, FunctionBaseId{c}, FunctionBaseId{k + c}, v);
			}
		}
		// Case l2 > l: compute for all (k,k2):
		for (ProcessId l2 = l; l2 < nb_processes; ++l2) {
			// Compute G_{l,l2,0,0} and copy to G_{l,l2,c,c}.
			{
				const auto v = G_ll2kk2 (l, l2, FunctionBaseId{0}, FunctionBaseId{0});
				for (size_t c = 0; c < base_size; ++c) {
					g.set_G (l, l2, FunctionBaseId{c}, FunctionBaseId{c}, v);
				}
			}
			/* for k in [1,K[:
			 * Compute G_{l,l2,0,k} and copy to G_{l,l2,c,k+c}.
			 * Compute G_{l,l2,k,0} and copy to G_{l,l2,k+c,c}.
			 */
			for (FunctionBaseId k = 1; k < base_size; ++k) {
				const auto v_0k = G_ll2kk2 (l, l2, FunctionBaseId{0}, k);
				const auto v_k0 = G_ll2kk2 (l, l2, k, FunctionBaseId{0});
				for (size_t c = 0; k + c < base_size; ++c) {
					g.set_G (l, l2, FunctionBaseId{c}, FunctionBaseId{k + c}, v_0k);
					g.set_G (l, l2, FunctionBaseId{k + c}, FunctionBaseId{c}, v_k0);
				}
			}
		}
	}
	return g;
}
inline CommonIntermediateValues compute_intermediate_values (const ProcessesRegionData & /*unused*/,
                                                             const HistogramBase & base,
                                                             const PointAndKernelData & data) {
	const auto nb_regions = data.nb_regions ();
	using B_G = typename CommonIntermediateValues::B_G;
	std::vector<B_G> b_g_by_region;
	b_g_by_region.reserve (nb_regions);
	for (RegionId r = 0; r < nb_regions; ++r) {
		fmt::print (stderr, "Region {}...\n", r);
		b_g_by_region.emplace_back (B_G{compute_b (data.processes_data_for_region (r), base),
		                                compute_g (data.processes_data_for_region (r), base)});
	}

	// B_hat too complex to compute
	Matrix_M_MK1 b_hat (data.nb_processes (), base.base_size);
	b_hat.inner.setZero ();

	return {std::move (b_g_by_region), std::move (b_hat)};
}

/******************************************************************************
 * Computations common to all cases.
 */

// This struct contains computed values specific to the chosen base and kernel.
struct CommonIntermediateValues {
	struct B_G {
		Matrix_M_MK1 b;
		MatrixG g;
	};
	std::vector<B_G> b_g_by_region;
	Matrix_M_MK1 b_hat;
};

template <typename Base, typename Kernels>
inline CommonIntermediateValues compute_intermediate_values (const DataByProcessRegion<SortedVec<Point>> & points,
                                                             const Base & base, const Kernels & kernels) {
	const auto nb_regions = points.nb_regions ();
	using B_G = typename CommonIntermediateValues::B_G;
	std::vector<B_G> b_g_by_region;
	b_g_by_region.reserve (nb_regions);
	for (RegionId r = 0; r < nb_regions; ++r) {
		b_g_by_region.emplace_back (B_G{compute_b (points.data_for_region (r), base, kernels),
		                                compute_g (points.data_for_region (r), base, kernels)});
	}
	return {std::move (b_g_by_region), compute_b_hat (points, base, kernels)};
}

struct LassoParameters {
	Matrix_M_MK1 sum_of_b;
	MatrixG sum_of_g;
	Matrix_M_MK1 d;
};

inline LassoParameters compute_lasso_parameters (const CommonIntermediateValues & values, double gamma) {
	const auto check_b_g = [](const Eigen::MatrixXd & m, const string_view what, RegionId r) {
		if (!m.allFinite ()) {
#ifndef NDEBUG
			// Print matrix in debug mode
			fmt::print (stderr, "##### Bad values for {}[region={:2}] #####\n", what, r);
			fmt::print (stderr, "{}\n", m);
			fmt::print (stderr, "#########################################\n");
#endif
			throw std::runtime_error (fmt::format ("{}[region={}] has non finite values", what, r));
		}
	};

	const auto nb_regions = values.b_g_by_region.size ();
	const auto nb_processes = values.b_hat.nb_processes;
	const auto base_size = values.b_hat.base_size;

	/* Generate values combining all regions:
	 * - sums of B and G.
	 * - V_hat = 1/R^2 * sum_r B[r]^2 (component-wise)
	 */
	Matrix_M_MK1 sum_of_b (nb_processes, base_size);
	MatrixG sum_of_g (nb_processes, base_size);
	Matrix_M_MK1 v_hat_r2 (nb_processes, base_size); // V_hat * R^2

	sum_of_b.inner.setZero ();
	sum_of_g.inner.setZero ();
	v_hat_r2.m_lk_values ().setZero ();

	for (RegionId r = 0; r < nb_regions; ++r) {
		const auto & v = values.b_g_by_region[r];
		check_b_g (v.b.inner, "B", r);
		check_b_g (v.g.inner, "G", r);
		sum_of_b.inner += v.b.inner;
		v_hat_r2.m_lk_values ().array () += v.b.m_lk_values ().array ().square ();
		sum_of_g.inner += v.g.inner;
	}

	/* Compute D, the penalty for the lassoshooting.
	 * V_hat_mkl is computed without dividing by R^2 so add the division to factor.
	 */
	const auto log_factor = std::log (nb_processes + nb_processes * nb_processes * base_size);
	const auto v_hat_factor = (1. / double(nb_regions * nb_regions)) * 2. * gamma * log_factor;
	const auto b_hat_factor = gamma * log_factor / 3.;

#ifndef NDEBUG
	// Compare v_hat and b_hat parts of d.
	{
		auto v_hat_part = (v_hat_factor * v_hat_r2.m_lk_values ().array ()).sqrt ();
		auto b_hat_part = b_hat_factor * values.b_hat.m_lk_values ().array ();
		const Eigen::IOFormat format (3, 0, "\t"); // 3 = precision in digits, this is enough
		fmt::print (stderr, "##### d = v_hat_part + b_hat_part: value of v_hat_part / b_hat_part #####\n");
		fmt::print (stderr, "{}\n", (v_hat_part / b_hat_part).format (format));
	}
#endif

	Matrix_M_MK1 d (nb_processes, base_size);
	d.m_0_values ().setZero (); // No penalty for constant component of estimators
	d.m_lk_values () =
	    (v_hat_factor * v_hat_r2.m_lk_values ().array ()).sqrt () + b_hat_factor * values.b_hat.m_lk_values ().array ();

	return {std::move (sum_of_b), std::move (sum_of_g), std::move (d)};
}

#ifndef NDEBUG
#include <Eigen/LU> // We need extra Eigen includes for matrix inversion.
#endif

inline Matrix_M_MK1 compute_estimated_a_with_lasso (const LassoParameters & p) {
	assert (p.sum_of_b.inner.allFinite ());
	assert (p.sum_of_g.inner.allFinite ());
	assert (p.d.inner.allFinite ());
#ifndef NDEBUG
	fmt::print (stderr, "############################### sum B ###################################\n");
	fmt::print (stderr, "{}\n", p.sum_of_b.inner);
	fmt::print (stderr, "############################### sum G ###################################\n");
	fmt::print (stderr, "{}\n", p.sum_of_g.inner);
	fmt::print (stderr, "################################# D #####################################\n");
	fmt::print (stderr, "{}\n", p.d.inner);
	fmt::print (stderr, "############################### G^-1*B ##################################\n");
	const auto inverse_g = p.sum_of_g.inner.fullPivLu ();
	if (inverse_g.isInvertible ()) {
		const Eigen::MatrixXd solution = inverse_g.solve (p.sum_of_b.inner);
		if ((p.sum_of_g.inner * solution).isApprox (p.sum_of_b.inner)) {
			fmt::print (stderr, "{}\n", solution);
		} else {
			fmt::print (stderr, "G * a = B has no solution\n");
		}
	} else {
		fmt::print (stderr, "G is not invertible\n");
	}
	fmt::print (stderr, "#########################################################################\n");
#endif
	const auto nb_processes = p.sum_of_b.nb_processes;
	const auto base_size = p.sum_of_b.base_size;
	Matrix_M_MK1 a (nb_processes, base_size);
	for (ProcessId m = 0; m < nb_processes; ++m) {
		const double lambda = 1.;
		a.values_for_m (m) = lassoshooting (p.sum_of_g.inner, p.sum_of_b.values_for_m (m), p.d.values_for_m (m), lambda);
	}
	return a;
}
