#pragma once
// Shape primitives specific to goodness computation

#include "shape.h"
#include "utils.h"

using shape::Indicator;
using shape::ShiftedPoints;

/* Tool to compute int_0^t sum_{y in points} indicator(x - y) dx.
 *
 * The strategy is similar to sup sum_{y in points} indicator(x - y) in the inferrence.
 * sum_{y in points} indicator(x - y) is a piecewise constant function.
 * x in interval for y <=> left <= x - y <= right <=> left + y <= x <= right + y.
 * The function changes value at {y + left / y in points}, {y + right / y in points}.
 *
 * We need to compute these integrals for all {x_m in N_m}.
 * An efficient way is to incrementally compute the integral for increasing x_m.
 * This way intermediate results for an x_m can be reused for the next one.
 * This makes the overall complexity proportional to max(|N_m|) instead of |N_m|^2.
 * A downside is the weird API.
 */
class IntegralSumPhiK {
  private:
    double integral;
    Point t;
    ShiftedPoints left_bounds;
    ShiftedPoints right_bounds;

  public:
    // An integral does not care for open/closed bounds, so the bound type is discarded
    template <Bound lb, Bound rb> IntegralSumPhiK(const SortedVec<Point> & points, const Interval<lb, rb> & interval)
        : integral(0.),
          t(std::numeric_limits<Point>::lowest()),
          left_bounds(points, interval.left),
          right_bounds(points, interval.right) {}

    // Zero integral. Used to start integrating from 0 instead of -inf.
    void set_value(double v) { integral = v; }

    // Incrementally integrate the sum from current t up to new_t, return new value.
    double value_at_t(Point new_t) {
        assert(new_t > t);
        while(true) {
            const Point next_change = std::min(left_bounds.point(), right_bounds.point());
            if(next_change == ShiftedPoints::inf) {
                break; // No more points to process.
            }
            const Point x = std::min(next_change, new_t);
            // The sum of indicator at x is the number of entered intervals minus the number of exited intervals.
            // Thus the sum is the difference between the indexes of the left bound iterator and the right one.
            assert(left_bounds.index() >= right_bounds.index());
            const auto sum_value_at_x = PointSpace(left_bounds.index() - right_bounds.index());
            integral += sum_value_at_x * (x - t);
            t = x;
            left_bounds.next_if_equal(x);
            right_bounds.next_if_equal(x);
            if(x == new_t) {
                break;
            }
        }
        return integral;
    }
};

/* Compute the set of lambda_hat_m for all x_m in N_m.
 *
 * lambda_hat_m(t) = sum_lk estimated_a_{m,l,k} int_0^t sum_{x_l in N_l} phi_k(x - x_l) dx.
 */
inline std::vector<double> compute_lambda_hat_m_for_all_Nm(
    span<const SortedVec<PointSpace>> points,
    std::size_t m,
    const HistogramBase & base,
    const Matrix_M_MK1 & estimated_a) {
    // Checks
    const auto nb_processes = estimated_a.nb_processes;
    const auto base_size = estimated_a.base_size;
    assert(points.size() == nb_processes);
    assert(base.base_size == base_size);
    assert(m < nb_processes);
    // Create list of {a(m,lk), integral(l,k)}, ignoring cases where a(m,l,k)==0
    struct WeightedIntegral {
        double weight;
        IntegralSumPhiK integral;
    };
    auto weighted_integrals = std::vector<WeightedIntegral>();
    for(ProcessId l = 0; l < nb_processes; l += 1) {
        for(FunctionBaseId k = 0; k < base_size; k += 1) {
            const double a = estimated_a.get_lk(m, l, k);
            if(a != 0.) {
                // Integrate up to 0. then reset accumulated value ; ready to start integration from 0.
                auto integral = IntegralSumPhiK(points[l], base.histogram(k).interval);
                integral.value_at_t(0.);
                integral.set_value(0.);
                weighted_integrals.push_back(WeightedIntegral{a * base.normalization_factor, std::move(integral)});
            }
        }
    }
    // Compute lambda_hat(x_m in N_m) incrementally. This relies on points[m] being sorted in increasing order.
    auto lambda_hat_for_x_m = std::vector<double>();
    lambda_hat_for_x_m.reserve(points[m].size());
    for(const Point x_m : points[m]) {
        double sum_a_integrals_indicator = 0.;
        for(auto & weighted_integral : weighted_integrals) {
            sum_a_integrals_indicator += weighted_integral.weight * weighted_integral.integral.value_at_t(x_m);
        }
        lambda_hat_for_x_m.push_back(sum_a_integrals_indicator);
    }
    return lambda_hat_for_x_m;
}

/* Compute one lambda_hat_m (used for t=tmax)
 *
 * lambda_hat_m(t) = sum_lk estimated_a_{m,l,k} int_0^t sum_{x_l in N_l} phi_k(x - x_l) dx.
 */
inline double compute_lambda_hat_m(
    span<const SortedVec<PointSpace>> points,
    std::size_t m,
    const HistogramBase & base,
    const Matrix_M_MK1 & estimated_a,
    Point t) {
    // Checks
    const auto nb_processes = estimated_a.nb_processes;
    const auto base_size = estimated_a.base_size;
    assert(points.size() == nb_processes);
    assert(base.base_size == base_size);
    assert(m < nb_processes);
    if (t <= 0.) {
        // Integrals from 0 to t.
        return 0.;
    }
    // Create list of {a(m,lk), integral(l,k)}, ignoring cases where a(m,l,k)==0
    double sum_a_integrals = 0.;
    for(ProcessId l = 0; l < nb_processes; l += 1) {
        for(FunctionBaseId k = 0; k < base_size; k += 1) {
            const double a = estimated_a.get_lk(m, l, k);
            if(a != 0.) {
                // Integrate up to 0. then reset accumulated value ; ready to start integration from 0.
                auto integral = IntegralSumPhiK(points[l], base.histogram(k).interval);
                integral.value_at_t(0.);
                integral.set_value(0.);
                sum_a_integrals += a * integral.value_at_t(t);
            }
        }
    }
    return sum_a_integrals * base.normalization_factor;
}