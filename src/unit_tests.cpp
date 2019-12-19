// First line
// Second line
// WHAT IS ABOVE THIS LINE IS USED BY TESTS AND MUST NOT BE CHANGED !
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "external/doctest.h"

#include <cstdio>

#include "command_line.h"
#include "computations.h"
#include "input.h"
#include "shape.h"
#include "utils.h"

template <typename T> static doctest::String toString(const std::vector<T> & v) {
    using doctest::toString;
    doctest::String s = "Vec{";
    if(v.size() > 0) {
        s += toString(v[0]);
    }
    for(std::size_t i = 1; i < v.size(); ++i) {
        s += toString(", ") + toString(v[i]);
    }
    return s + toString("}");
}
template <Bound lb, Bound rb> static doctest::String toString(Interval<lb, rb> i) {
    using doctest::toString;
    doctest::String s = "Interval{";
    s += toString(i.left);
    s += ",";
    s += toString(i.right);
    s += "}";
    return s;
}

/******************************************************************************
 * Types definitions
 */
TEST_SUITE("types") {
    TEST_CASE("haar_base_k_mapping") {
        const auto base = HaarBase(4, 100.);
        // Some values
        const auto sp_0 = base.scale_and_position(0);
        CHECK(sp_0.scale == 0);
        CHECK(sp_0.position == 0);
        const auto sp_1 = base.scale_and_position(1);
        CHECK(sp_1.scale == 1);
        CHECK(sp_1.position == 0);
        const auto sp_2 = base.scale_and_position(2);
        CHECK(sp_2.scale == 1);
        CHECK(sp_2.position == 1);
        const auto sp_3 = base.scale_and_position(3);
        CHECK(sp_3.scale == 2);
        CHECK(sp_3.position == 0);
        // Check bijection on all values.
        for(FunctionBaseId k = 0; k < base.base_size(); ++k) {
            const auto sp = base.scale_and_position(k);
            const auto k2 = base.base_id(sp.scale, sp.position);
            CHECK(k == k2);
        }
    }
}

/******************************************************************************
 * Shape tests.
 */
template <Bound lb, Bound rb> struct IndicatorLike {
    Interval<lb, rb> interval;
    Interval<lb, rb> non_zero_domain() const { return interval; }
    double operator()(Point x) const {
        if(interval.contains(x)) {
            return 1.;
        } else {
            return 0.;
        }
    }
};
struct TriangleShape {
    double size;
    Interval<Bound::Closed, Bound::Closed> non_zero_domain() const { return {0., size}; }
    double operator()(Point x) const {
        if(non_zero_domain().contains(x)) {
            return x;
        } else {
            return 0.;
        }
    }
};
template <Bound lb, Bound rb> struct ScaledIndicatorLike {
    Interval<lb, rb> interval;
    double scale;
    Interval<lb, rb> non_zero_domain() const { return interval; }
    double operator()(Point x) const {
        if(interval.contains(x)) {
            return scale;
        } else {
            return 0.;
        }
    }
};

static double raw_evaluate_at(shape::Polynomial p, Point x) {
    // ignore nzd, used to test continuity between convolution components
    return shape::compute_polynom_value(x - p.origin, p.coefficients);
}
static shape::Polynom<Bound::Closed, Bound::Closed> positive_triangle(PointSpace size) {
    return {
        {0., size},
        {size / 2., 1.},
    };
}
static shape::Polynom<Bound::Closed, Bound::Closed> negative_triangle(PointSpace size) {
    return {
        {-size, 0.},
        {size / 2., -1.},
    };
}

TEST_SUITE("shape") {
    using namespace shape;

    TEST_CASE("reverse") {
        auto p = Polynom<Bound::Closed, Bound::Closed>{
            {0., 10.},
            {1., -1., 1., -2.},
        };
        auto rev_p = reverse(p);
        for(int x = -15; x <= 15; x += 1) {
            CHECK(rev_p(x) == p(-x));
        }
    }
    TEST_CASE("trapezoids") {
        auto indicator = Indicator<Bound::Closed, Bound::Closed>{{-1., 1.}};
        auto indicator_shifted = Indicator<Bound::Closed, Bound::Closed>{{0., 2.}};
        auto indicator_large = Indicator<Bound::Closed, Bound::Closed>{{-2., 2.}};
        {
            auto degenerate_trapezoid = convolution(indicator, indicator);
            // Properties
            CHECK(degenerate_trapezoid.components.size() == 2); // Optimized
            const auto & up = degenerate_trapezoid.components[0];
            CHECK(up.interval == Interval<Bound::Open, Bound::Closed>{-2., 0.});
            CHECK(up.degree() == 1);
            CHECK(up.coefficients[1] == 1.);
            const auto & down = degenerate_trapezoid.components[1];
            CHECK(down.interval == Interval<Bound::Open, Bound::Closed>{0., 2.});
            CHECK(down.degree() == 1);
            CHECK(down.coefficients[1] == -1.);
            // Values
            CHECK(degenerate_trapezoid(-3) == 0);
            CHECK(degenerate_trapezoid(-2) == 0);
            CHECK(degenerate_trapezoid(-1) == 1);
            CHECK(degenerate_trapezoid(0) == 2);
            CHECK(degenerate_trapezoid(1) == 1);
            CHECK(degenerate_trapezoid(2) == 0);
            CHECK(degenerate_trapezoid(3) == 0);
            // Continuity
            CHECK(raw_evaluate_at(up, -2.) == 0.);
            CHECK(raw_evaluate_at(up, 0.) == 2.);
            CHECK(raw_evaluate_at(down, 0.) == 2.);
            CHECK(raw_evaluate_at(down, 2.) == 0.);
            // Cross correlation should be the same
            auto dt_cc = cross_correlation(indicator, indicator);
            CHECK(dt_cc(-2) == 0);
            CHECK(dt_cc(-1) == 1);
            CHECK(dt_cc(0) == 2);
            CHECK(dt_cc(1) == 1);
            CHECK(dt_cc(2) == 0);
        }
        {
            // Shifting
            auto trapezoid_1sa = convolution(indicator, indicator_shifted);
            CHECK(trapezoid_1sa.non_zero_domain() == Interval<Bound::Open, Bound::Closed>{-1., 3.});
            CHECK(trapezoid_1sa(-1) == 0);
            CHECK(trapezoid_1sa(0) == 1);
            CHECK(trapezoid_1sa(1) == 2);
            CHECK(trapezoid_1sa(2) == 1);
            CHECK(trapezoid_1sa(3) == 0);
            auto trapezoid_1sb = convolution(indicator_shifted, indicator);
            CHECK(trapezoid_1sb.non_zero_domain() == Interval<Bound::Open, Bound::Closed>{-1., 3.});
            CHECK(trapezoid_1sb(-1) == 0);
            CHECK(trapezoid_1sb(0) == 1);
            CHECK(trapezoid_1sb(1) == 2);
            CHECK(trapezoid_1sb(2) == 1);
            CHECK(trapezoid_1sb(3) == 0);
            auto trapezoid_2s = convolution(indicator_shifted, indicator_shifted);
            CHECK(trapezoid_2s.non_zero_domain() == Interval<Bound::Open, Bound::Closed>{0., 4.});
            CHECK(trapezoid_2s(0) == 0);
            CHECK(trapezoid_2s(1) == 1);
            CHECK(trapezoid_2s(2) == 2);
            CHECK(trapezoid_2s(3) == 1);
            CHECK(trapezoid_2s(4) == 0);
            // Cross correlation
            auto trapezoid_r1sa = cross_correlation(indicator, indicator_shifted);
            CHECK(trapezoid_r1sa.non_zero_domain() == Interval<Bound::Open, Bound::Closed>{-1., 3.});
            CHECK(trapezoid_r1sa(-1) == 0);
            CHECK(trapezoid_r1sa(0) == 1);
            CHECK(trapezoid_r1sa(1) == 2);
            CHECK(trapezoid_r1sa(2) == 1);
            CHECK(trapezoid_r1sa(3) == 0);
            auto trapezoid_r1sb = cross_correlation(indicator_shifted, indicator);
            CHECK(trapezoid_r1sb.non_zero_domain() == Interval<Bound::Open, Bound::Closed>{-3., 1.});
            CHECK(trapezoid_r1sb(-3) == 0);
            CHECK(trapezoid_r1sb(-2) == 1);
            CHECK(trapezoid_r1sb(-1) == 2);
            CHECK(trapezoid_r1sb(0) == 1);
            CHECK(trapezoid_r1sb(1) == 0);
            auto trapezoid_r2s = cross_correlation(indicator_shifted, indicator_shifted);
            CHECK(trapezoid_r2s.non_zero_domain() == Interval<Bound::Open, Bound::Closed>{-2., 2.});
            CHECK(trapezoid_r2s(-2) == 0);
            CHECK(trapezoid_r2s(-1) == 1);
            CHECK(trapezoid_r2s(0) == 2);
            CHECK(trapezoid_r2s(1) == 1);
            CHECK(trapezoid_r2s(2) == 0);
        }
        {
            // Full trapezoids
            auto trapezoid = convolution(indicator, indicator_large);
            CHECK(trapezoid.non_zero_domain() == Interval<Bound::Open, Bound::Closed>{-3., 3.});
            CHECK(trapezoid(-4) == 0);
            CHECK(trapezoid(-3) == 0);
            CHECK(trapezoid(-2) == 1);
            CHECK(trapezoid(-1) == 2);
            CHECK(trapezoid(0) == 2);
            CHECK(trapezoid(1) == 2);
            CHECK(trapezoid(2) == 1);
            CHECK(trapezoid(3) == 0);
            CHECK(trapezoid(4) == 0);
            auto r_trapezoid = cross_correlation(indicator, indicator_large);
            for(int x = -4; x <= 4; x += 1) {
                CHECK(trapezoid(x) == r_trapezoid(x));
            }
        }
    }
    TEST_CASE("known_convolutions") {
        {
            // square * pos_tri, Case l <= c
            const auto interval = Indicator<Bound::Closed, Bound::Closed>{{-1., 1.}};
            const auto tri = positive_triangle(3.);
            const auto conv = convolution(interval, tri);
            CHECK(conv.non_zero_domain() == Interval<Bound::Open, Bound::Closed>{-1, 3 + 1});
            CHECK(conv(-2) == 0);
            CHECK(conv(-1) == 0); // f(-l/2) == 0
            CHECK(conv(0) > conv(-1));
            CHECK(conv(1) > conv(0));
            CHECK(conv(1) == 2 * 2 / 2); // f(l/2) = l^2/2
            CHECK(conv(2) > conv(1));
            CHECK(conv(2) == 3 * 2 - 2 * 2 / 2); // f(c-l/2) = cl - l^2/2
            CHECK(conv(3) < conv(2));
            CHECK(conv(4) < conv(3));
            CHECK(conv(4) == 0);
            CHECK(conv(5) == 0);
        }
        {
            // square * pos_tri, Case l >= c
            const auto interval = Indicator<Bound::Closed, Bound::Closed>{{-1., 1.}};
            const auto tri = positive_triangle(1.);
            const auto conv = convolution(interval, tri);
            CHECK(conv.non_zero_domain() == Interval<Bound::Open, Bound::Closed>{-1, 1 + 1});
            CHECK(conv(-2) == 0);
            CHECK(conv(-1) == 0); // f(-l/2) == 0
            CHECK(conv(0) > conv(-1));
            CHECK(conv(0) == 1. * 1. / 2.); // f(c-l/2) = c^2/2
            CHECK(conv(1) == 1. * 1. / 2.); // f(l/2) = c^2/2
            CHECK(conv(2) < conv(1));
            CHECK(conv(2) == 0);
            CHECK(conv(3) == 0);
        }
        {
            // pos_tri * pos_tri
            // 2 symmetrical cases: use a=3,b=4 so that the max of the shape is at 5.
            const auto a_tri = positive_triangle(3);
            const auto b_tri = positive_triangle(4);
            const auto conv = convolution(a_tri, b_tri);
            CHECK(conv.non_zero_domain() == Interval<Bound::Open, Bound::Closed>{0, 3 + 4});
            CHECK(conv(-1) == 0);
            CHECK(conv(0) == 0);
            CHECK(conv(1) > conv(0));
            CHECK(conv(2) > conv(1));
            CHECK(conv(3) > conv(2));
            CHECK(conv(3) == doctest::Approx(3. * 3. * 3. / 6.));
            CHECK(conv(4) > conv(3));
            CHECK(conv(4) == doctest::Approx(3. * 3. * (3. * 4. - 2. * 3.) / 6.));
            CHECK(conv(5) > conv(4));
            CHECK(conv(6) < conv(5));
            CHECK(conv(7) < conv(6));
            CHECK(conv(7) == doctest::Approx(0));
            CHECK(conv(8) == 0);
        }
        {
            // neg_tri * pos_tri
            // 2 cases with results time reverted from the other.
            const auto neg_tri = negative_triangle(2);
            const auto pos_tri = positive_triangle(3);
            const auto conv = convolution(neg_tri, pos_tri);
            CHECK(conv.non_zero_domain() == Interval<Bound::Open, Bound::Closed>{-2, 3});
            CHECK(conv(-3) == 0);
            CHECK(conv(-2) == 0);
            CHECK(conv(-1) > conv(-2));
            CHECK(conv(0) > conv(-1));
            CHECK(conv(0) == doctest::Approx(2. * 2. * 2. / 3.));
            CHECK(conv(1) > conv(0));
            CHECK(conv(1) == doctest::Approx(2. * 2. * (3. * 3. - 2.) / 6.));
            CHECK(conv(2) < conv(1));
            CHECK(conv(3) < conv(2));
            CHECK(conv(3) == 0);
            CHECK(conv(4) == 0);
        }
    }
    TEST_CASE("generic_convolution_properties") {
        // Check operations at high degree
        auto p = Polynom<Bound::Closed, Bound::Closed>{
            {-8., 8.},
            {64., 0., -1.},
        };
        auto q = Polynom<Bound::Closed, Bound::Closed>{
            {-10., 10.},
            {0., -100., 0., 1.},
        };
        {
            auto conv = convolution(p, q);
            CHECK(conv.components.size() == 3);
            const auto & left = conv.components[0];
            const auto & center = conv.components[1];
            const auto & right = conv.components[2];
            // Continuity
            CHECK(raw_evaluate_at(left, left.interval.left) == doctest::Approx(0.));
            CHECK(
                raw_evaluate_at(left, left.interval.right) ==
                doctest::Approx(raw_evaluate_at(center, center.interval.left)));
            CHECK(
                raw_evaluate_at(center, center.interval.right) ==
                doctest::Approx(raw_evaluate_at(right, right.interval.left)));
            CHECK(raw_evaluate_at(right, right.interval.right) == doctest::Approx(0.));
        }
        {
            auto cross = cross_correlation(p, q);
            CHECK(cross.components.size() == 3);
            const auto & left = cross.components[0];
            const auto & center = cross.components[1];
            const auto & right = cross.components[2];
            // Continuity
            CHECK(raw_evaluate_at(left, left.interval.left) == doctest::Approx(0.));
            CHECK(
                raw_evaluate_at(left, left.interval.right) ==
                doctest::Approx(raw_evaluate_at(center, center.interval.left)));
            CHECK(
                raw_evaluate_at(center, center.interval.right) ==
                doctest::Approx(raw_evaluate_at(right, right.interval.left)));
            CHECK(raw_evaluate_at(right, right.interval.right) == doctest::Approx(0.));
        }
    }
    TEST_CASE("integral") {
        // Simple constant function
        const auto indicator = Indicator<Bound::Closed, Bound::Closed>{{0., 42.}};
        CHECK(integral(indicator) == doctest::Approx(42.));
        // Linear function with equal time negative and positive
        const auto linear_with_0_integral = Polynom<Bound::Closed, Bound::Closed>{
            {0., 2.},
            {0., 1.},
        };
        CHECK(integral(linear_with_0_integral) == doctest::Approx(0.));
        // Degree 2 simple function, integral = x^3/3
        const auto degree_2 = Polynom<Bound::Closed, Bound::Closed>{
            {-10., 10.},
            {0., 0., 1.},
        };
        CHECK(integral(degree_2) == doctest::Approx(2. * (10. * 10. * 10.) / 3.));
    }
    TEST_CASE("indicator_approximation") {
        // Trivial case
        const auto indicator = Indicator<Bound::Closed, Bound::Closed>{{0., 42.}};
        const Indicator<Bound::Closed, Bound::Closed> indicator_a = indicator_approximation(indicator);
        CHECK(indicator_a.interval == indicator.interval);
        // Linear function, "climbing" triangle of size 2.
        const auto linear = positive_triangle(2.);
        const auto linear_a = indicator_approximation(linear);
        CHECK(linear_a.non_zero_domain() == linear.non_zero_domain());
        CHECK(linear_a(0.) == doctest::Approx(1.));
        // Sum of linears : positive then negative slopes of lengths 2
        const auto mountain = Add<std::vector<Polynom<Bound::Open, Bound::Closed>>>{{
            Polynom<Bound::Open, Bound::Closed>{{0., 2.}, {1., 1.}},  // Positive slope
            Polynom<Bound::Open, Bound::Closed>{{2., 4.}, {1., -1.}}, // Negative slope
        }};
        const auto mountain_a = indicator_approximation(mountain);
        CHECK(mountain_a.non_zero_domain() == Interval<Bound::Open, Bound::Closed>{0., 4.});
        CHECK(mountain_a(1.) == doctest::Approx(1.));
    }
    TEST_CASE("sum_shape_point_differences_generic") {
        // Tested functions
        const auto indicator_oo = IndicatorLike<Bound::Open, Bound::Open>{{-1, 1}};
        const auto indicator_oc = IndicatorLike<Bound::Open, Bound::Closed>{{-1, 1}};
        const auto indicator_co = IndicatorLike<Bound::Closed, Bound::Open>{{-1, 1}};
        const auto indicator_cc = IndicatorLike<Bound::Closed, Bound::Closed>{{-1, 1}};

        // Tested point sets
        const auto empty = SortedVec<Point>::from_sorted({});
        const auto zero = SortedVec<Point>::from_sorted({0});
        const auto one = SortedVec<Point>::from_sorted({1});
        const auto all_near_zero = SortedVec<Point>::from_sorted({-4, -3, -2, -1, 0, 1, 2, 3, 4});

        // Should be zero due to emptyset
        CHECK(sum_shape_point_differences(empty, empty, indicator_oo) == 0);
        CHECK(sum_shape_point_differences(empty, zero, indicator_oo) == 0);
        CHECK(sum_shape_point_differences(zero, empty, indicator_oo) == 0);

        CHECK(sum_shape_point_differences(empty, empty, indicator_oc) == 0);
        CHECK(sum_shape_point_differences(empty, zero, indicator_oc) == 0);
        CHECK(sum_shape_point_differences(zero, empty, indicator_oc) == 0);

        CHECK(sum_shape_point_differences(empty, empty, indicator_co) == 0);
        CHECK(sum_shape_point_differences(empty, zero, indicator_co) == 0);
        CHECK(sum_shape_point_differences(zero, empty, indicator_co) == 0);

        CHECK(sum_shape_point_differences(empty, empty, indicator_cc) == 0);
        CHECK(sum_shape_point_differences(empty, zero, indicator_cc) == 0);
        CHECK(sum_shape_point_differences(zero, empty, indicator_cc) == 0);

        // Only one same point in both sets
        CHECK(sum_shape_point_differences(zero, zero, indicator_oo) == 1);
        CHECK(sum_shape_point_differences(zero, zero, indicator_oc) == 1);
        CHECK(sum_shape_point_differences(zero, zero, indicator_cc) == 1);
        CHECK(sum_shape_point_differences(zero, zero, indicator_cc) == 1);

        // Two sets with one point, one diff = 1 for (one, zero), -1 for (zero, one)
        CHECK(sum_shape_point_differences(one, zero, indicator_oo) == 0);
        CHECK(sum_shape_point_differences(zero, one, indicator_oo) == 0);

        CHECK(sum_shape_point_differences(one, zero, indicator_oc) == 1);
        CHECK(sum_shape_point_differences(zero, one, indicator_oc) == 0);

        CHECK(sum_shape_point_differences(one, zero, indicator_co) == 0);
        CHECK(sum_shape_point_differences(zero, one, indicator_co) == 1);

        CHECK(sum_shape_point_differences(one, zero, indicator_cc) == 1);
        CHECK(sum_shape_point_differences(zero, one, indicator_cc) == 1);

        // Single point with all points near zero : set of diffs = all_near_zero
        CHECK(sum_shape_point_differences(zero, all_near_zero, indicator_oo) == 1);
        CHECK(sum_shape_point_differences(all_near_zero, zero, indicator_oo) == 1);

        CHECK(sum_shape_point_differences(zero, all_near_zero, indicator_oc) == 2);
        CHECK(sum_shape_point_differences(all_near_zero, zero, indicator_oc) == 2);

        CHECK(sum_shape_point_differences(zero, all_near_zero, indicator_co) == 2);
        CHECK(sum_shape_point_differences(all_near_zero, zero, indicator_co) == 2);

        CHECK(sum_shape_point_differences(zero, all_near_zero, indicator_cc) == 3);
        CHECK(sum_shape_point_differences(all_near_zero, zero, indicator_cc) == 3);

        // Multiple points with all points near zero
        const auto some_points = SortedVec<Point>::from_sorted({-3, 0, 3});
        CHECK(sum_shape_point_differences(some_points, all_near_zero, indicator_oo) == 3);
        CHECK(sum_shape_point_differences(all_near_zero, some_points, indicator_oo) == 3);
        CHECK(sum_shape_point_differences(some_points, some_points, indicator_oo) == 3);

        CHECK(sum_shape_point_differences(some_points, all_near_zero, indicator_oc) == 6);
        CHECK(sum_shape_point_differences(all_near_zero, some_points, indicator_oc) == 6);
        CHECK(sum_shape_point_differences(some_points, some_points, indicator_oc) == 3);

        CHECK(sum_shape_point_differences(some_points, all_near_zero, indicator_co) == 6);
        CHECK(sum_shape_point_differences(all_near_zero, some_points, indicator_co) == 6);
        CHECK(sum_shape_point_differences(some_points, some_points, indicator_co) == 3);

        CHECK(sum_shape_point_differences(some_points, all_near_zero, indicator_cc) == 9);
        CHECK(sum_shape_point_differences(all_near_zero, some_points, indicator_cc) == 9);
        CHECK(sum_shape_point_differences(some_points, some_points, indicator_cc) == 3);

        // Tests values for non indicator shape
        const auto triangle = TriangleShape{4.};
        CHECK(sum_shape_point_differences(empty, zero, triangle) == 0);
        CHECK(sum_shape_point_differences(zero, empty, triangle) == 0);
        CHECK(sum_shape_point_differences(zero, zero, triangle) == 0);
        CHECK(sum_shape_point_differences(one, zero, triangle) == 1);
        CHECK(sum_shape_point_differences(all_near_zero, zero, triangle) == 10); // 0+1+2+3+4
        CHECK(sum_shape_point_differences(zero, all_near_zero, triangle) == 10); // 0+1+2+3+4
        CHECK(sum_shape_point_differences(all_near_zero, one, triangle) == 6);   // 0+1+2+3
        CHECK(sum_shape_point_differences(one, all_near_zero, triangle) == 10);  // 0+1+2+3+4
    }
    TEST_CASE("sum_shape_point_differences_squared_generic") {
        // Tested functions
        const auto indicator_oo = IndicatorLike<Bound::Open, Bound::Open>{{-1, 1}};
        const auto indicator_oc = IndicatorLike<Bound::Open, Bound::Closed>{{-1, 1}};
        const auto indicator_co = IndicatorLike<Bound::Closed, Bound::Open>{{-1, 1}};
        const auto indicator_cc = IndicatorLike<Bound::Closed, Bound::Closed>{{-1, 1}};

        // Tested point sets
        const auto empty = SortedVec<Point>::from_sorted({});
        const auto zero = SortedVec<Point>::from_sorted({0});
        const auto one = SortedVec<Point>::from_sorted({1});
        const auto all_near_zero = SortedVec<Point>::from_sorted({-4, -3, -2, -1, 0, 1, 2, 3, 4});

        // Should be zero due to emptyset
        CHECK(sum_shape_point_differences_squared(empty, empty, indicator_oo) == 0);
        CHECK(sum_shape_point_differences_squared(empty, zero, indicator_oo) == 0);
        CHECK(sum_shape_point_differences_squared(zero, empty, indicator_oo) == 0);

        CHECK(sum_shape_point_differences_squared(empty, empty, indicator_oc) == 0);
        CHECK(sum_shape_point_differences_squared(empty, zero, indicator_oc) == 0);
        CHECK(sum_shape_point_differences_squared(zero, empty, indicator_oc) == 0);

        CHECK(sum_shape_point_differences_squared(empty, empty, indicator_co) == 0);
        CHECK(sum_shape_point_differences_squared(empty, zero, indicator_co) == 0);
        CHECK(sum_shape_point_differences_squared(zero, empty, indicator_co) == 0);

        CHECK(sum_shape_point_differences_squared(empty, empty, indicator_cc) == 0);
        CHECK(sum_shape_point_differences_squared(empty, zero, indicator_cc) == 0);
        CHECK(sum_shape_point_differences_squared(zero, empty, indicator_cc) == 0);

        // Only one same point in both sets
        CHECK(sum_shape_point_differences_squared(zero, zero, indicator_oo) == 1 * 1);
        CHECK(sum_shape_point_differences_squared(zero, zero, indicator_oc) == 1 * 1);
        CHECK(sum_shape_point_differences_squared(zero, zero, indicator_cc) == 1 * 1);
        CHECK(sum_shape_point_differences_squared(zero, zero, indicator_cc) == 1 * 1);

        // Two sets with one point, one diff = 1 for (one, zero), -1 for (zero, one)
        CHECK(sum_shape_point_differences_squared(one, zero, indicator_oo) == 0);
        CHECK(sum_shape_point_differences_squared(zero, one, indicator_oo) == 0);

        CHECK(sum_shape_point_differences_squared(one, zero, indicator_oc) == 1 * 1);
        CHECK(sum_shape_point_differences_squared(zero, one, indicator_oc) == 0);

        CHECK(sum_shape_point_differences_squared(one, zero, indicator_co) == 0);
        CHECK(sum_shape_point_differences_squared(zero, one, indicator_co) == 1 * 1);

        CHECK(sum_shape_point_differences_squared(one, zero, indicator_cc) == 1 * 1);
        CHECK(sum_shape_point_differences_squared(zero, one, indicator_cc) == 1 * 1);

        // Single point with all points near zero : set of diffs = all_near_zero
        CHECK(sum_shape_point_differences_squared(zero, all_near_zero, indicator_oo) == 1 * 1);
        CHECK(sum_shape_point_differences_squared(all_near_zero, zero, indicator_oo) == 1 * 1);

        CHECK(sum_shape_point_differences_squared(zero, all_near_zero, indicator_oc) == 1 * (2 * 2));
        CHECK(sum_shape_point_differences_squared(all_near_zero, zero, indicator_oc) == 2 * 1);

        CHECK(sum_shape_point_differences_squared(zero, all_near_zero, indicator_co) == 1 * (2 * 2));
        CHECK(sum_shape_point_differences_squared(all_near_zero, zero, indicator_co) == 2 * 1);

        CHECK(sum_shape_point_differences_squared(zero, all_near_zero, indicator_cc) == 1 * (3 * 3));
        CHECK(sum_shape_point_differences_squared(all_near_zero, zero, indicator_cc) == 3 * 1);

        // Multiple points with all points near zero
        const auto some_points = SortedVec<Point>::from_sorted({-3, 0, 3});
        CHECK(sum_shape_point_differences_squared(some_points, all_near_zero, indicator_oo) == 3 * 1);
        CHECK(sum_shape_point_differences_squared(all_near_zero, some_points, indicator_oo) == 3 * 1);
        CHECK(sum_shape_point_differences_squared(some_points, some_points, indicator_oo) == 3 * 1);

        CHECK(sum_shape_point_differences_squared(some_points, all_near_zero, indicator_oc) == 3 * (2 * 2));
        CHECK(sum_shape_point_differences_squared(all_near_zero, some_points, indicator_oc) == 6 * 1);
        CHECK(sum_shape_point_differences_squared(some_points, some_points, indicator_oc) == 3 * 1);

        CHECK(sum_shape_point_differences_squared(some_points, all_near_zero, indicator_co) == 3 * (2 * 2));
        CHECK(sum_shape_point_differences_squared(all_near_zero, some_points, indicator_co) == 6 * 1);
        CHECK(sum_shape_point_differences_squared(some_points, some_points, indicator_co) == 3 * 1);

        CHECK(sum_shape_point_differences_squared(some_points, all_near_zero, indicator_cc) == 3 * (3 * 3));
        CHECK(sum_shape_point_differences_squared(all_near_zero, some_points, indicator_cc) == 9 * 1);
        CHECK(sum_shape_point_differences_squared(some_points, some_points, indicator_cc) == 3 * 1);

        // Tests values for non indicator shape
        const auto triangle = TriangleShape{4.};
        CHECK(sum_shape_point_differences_squared(empty, zero, triangle) == 0);
        CHECK(sum_shape_point_differences_squared(zero, empty, triangle) == 0);
        CHECK(sum_shape_point_differences_squared(zero, zero, triangle) == 0);
        CHECK(sum_shape_point_differences_squared(one, zero, triangle) == 1 * 1);
        CHECK(sum_shape_point_differences_squared(all_near_zero, zero, triangle) == 30);  // 1*0+1*1^2+1*2^2+1*3^2+1*4^2
        CHECK(sum_shape_point_differences_squared(zero, all_near_zero, triangle) == 100); // 1*(0+1+2+3+4)^2
    }
    TEST_CASE("sup_sum_shape_differences_to_points_indicators") {
        const auto vec_empty = SortedVec<Point>::from_sorted({});
        const auto vec_0_3_6 = SortedVec<Point>::from_sorted({0, 3, 6});
        const auto vec_0_2 = SortedVec<Point>::from_sorted({0, 2});
        const auto vec_0_1_2 = SortedVec<Point>::from_sorted({0, 1, 2});

        const auto indicator_oo = Indicator<Bound::Open, Bound::Open>{{0, 3}};
        const auto indicator_oc = Indicator<Bound::Open, Bound::Closed>{{0, 3}};
        const auto indicator_co = Indicator<Bound::Closed, Bound::Open>{{0, 3}};
        const auto indicator_cc = Indicator<Bound::Closed, Bound::Closed>{{0, 3}};

        CHECK(sup_sum_shape_differences_to_points(vec_empty, indicator_oo) == 0);
        CHECK(sup_sum_shape_differences_to_points(vec_empty, indicator_oc) == 0);
        CHECK(sup_sum_shape_differences_to_points(vec_empty, indicator_co) == 0);
        CHECK(sup_sum_shape_differences_to_points(vec_empty, indicator_cc) == 0);

        CHECK(sup_sum_shape_differences_to_points(vec_0_3_6, indicator_oo) == 1);
        CHECK(sup_sum_shape_differences_to_points(vec_0_3_6, indicator_oc) == 1);
        CHECK(sup_sum_shape_differences_to_points(vec_0_3_6, indicator_co) == 1);
        CHECK(sup_sum_shape_differences_to_points(vec_0_3_6, indicator_cc) == 2);

        CHECK(sup_sum_shape_differences_to_points(vec_0_2, indicator_oo) == 2);
        CHECK(sup_sum_shape_differences_to_points(vec_0_2, indicator_oc) == 2);
        CHECK(sup_sum_shape_differences_to_points(vec_0_2, indicator_co) == 2);
        CHECK(sup_sum_shape_differences_to_points(vec_0_2, indicator_cc) == 2);

        CHECK(sup_sum_shape_differences_to_points(vec_0_1_2, indicator_oo) == 3);
        CHECK(sup_sum_shape_differences_to_points(vec_0_1_2, indicator_oc) == 3);
        CHECK(sup_sum_shape_differences_to_points(vec_0_1_2, indicator_co) == 3);
        CHECK(sup_sum_shape_differences_to_points(vec_0_1_2, indicator_cc) == 3);
    }
    TEST_CASE("scale_optimisations") {
        // Check that scale optimisations are valid by comparing with a non optimized scaling
        const auto indicator = Indicator<Bound::Closed, Bound::Closed>{{0, 3}};
        const auto scaled_opt = scaled(2., indicator);
        const auto scaled_no_opt = ScaledIndicatorLike<Bound::Closed, Bound::Closed>{{0, 3}, 2.};

        const auto all_near_zero = SortedVec<Point>::from_sorted({-4, -3, -2, -1, 0, 1, 2, 3, 4});
        const auto zero = SortedVec<Point>::from_sorted({0});

        CHECK(
            sum_shape_point_differences(zero, all_near_zero, scaled_opt) ==
            sum_shape_point_differences(zero, all_near_zero, scaled_no_opt));
        CHECK(
            sum_shape_point_differences_squared(zero, all_near_zero, scaled_opt) ==
            sum_shape_point_differences_squared(zero, all_near_zero, scaled_no_opt));
    }
}

/******************************************************************************
 * Computations tests.
 */
TEST_SUITE("computations") {
    TEST_CASE("tmax") {
        const SortedVec<Point> one_array[] = {SortedVec<Point>::from_sorted({-1, 1})};
        CHECK(tmax(make_span(one_array)) == 2);
        CHECK(tmax(make_span(&one_array[0], 0)) == 0); // Empty
        const SortedVec<Point> two_arrays[] = {
            SortedVec<Point>::from_sorted({0, 42}),
            SortedVec<Point>::from_sorted({-1, 1}),
        };
        CHECK(tmax(make_span(two_arrays)) == 43);
        const SortedVec<Point> contains_empty[] = {
            SortedVec<Point>::from_sorted({0, 42}),
            SortedVec<Point>::from_sorted({}),
        };
        CHECK(tmax(make_span(contains_empty)) == 42);
        const SortedVec<Point> one_point[] = {SortedVec<Point>::from_sorted({1})};
        CHECK(tmax(make_span(one_point)) == 0);
    }
    /*TEST_CASE ("b_ml_histogram_counts_for_all_k_denormalized") {
            const auto base = HistogramBase{3, 1}; // K=3, delta=1, so intervals=]0,1] ]1,2] ]2,3]
            const auto empty = SortedVec<Point>::from_sorted ({});
            const auto point = SortedVec<Point>::from_sorted ({0});
            const auto points = SortedVec<Point>::from_sorted ({0, 1, 2});
            // Edge cases: empty
            CHECK (b_ml_histogram_counts_for_all_k_denormalized (empty, empty, base) == std::vector<int64_t>{0, 0, 0});
            CHECK (b_ml_histogram_counts_for_all_k_denormalized (point, empty, base) == std::vector<int64_t>{0, 0, 0});
            CHECK (b_ml_histogram_counts_for_all_k_denormalized (empty, point, base) == std::vector<int64_t>{0, 0, 0});
            CHECK (b_ml_histogram_counts_for_all_k_denormalized (points, empty, base) == std::vector<int64_t>{0, 0, 0});
            CHECK (b_ml_histogram_counts_for_all_k_denormalized (empty, points, base) == std::vector<int64_t>{0, 0, 0});
            // Non empty
            CHECK (b_ml_histogram_counts_for_all_k_denormalized (points, point, base) == std::vector<int64_t>{1, 1, 0});
            CHECK (b_ml_histogram_counts_for_all_k_denormalized (points, points, base) == std::vector<int64_t>{2, 1,
    0});
    }
    TEST_CASE ("g_ll2kk2_histogram_integral_denormalized") {
            const auto interval_0_1 = HistogramBase::Interval{0, 1}; // ]0,1]
            const auto interval_0_2 = HistogramBase::Interval{0, 2}; // ]0,2]
            const auto interval_2_4 = HistogramBase::Interval{2, 4}; // ]2,4]
            const auto empty = SortedVec<Point>::from_sorted ({});
            const auto point = SortedVec<Point>::from_sorted ({0});
            const auto points = SortedVec<Point>::from_sorted ({0, 1, 2});
            // Empty
            CHECK (g_ll2kk2_histogram_integral_denormalized (empty, interval_0_1, empty, interval_0_1) == 0);
            CHECK (g_ll2kk2_histogram_integral_denormalized (point, interval_0_1, empty, interval_0_1) == 0);
            CHECK (g_ll2kk2_histogram_integral_denormalized (empty, interval_0_1, point, interval_0_1) == 0);
            // With phi=indicator(]0,1]), phi(x - x_l) * phi(x - x_m) = if x_m == x_l then 1 else 0
            // Thus integral_x sum_{x_l,x_m} phi(x - x_m) phi(x - x_l) = sum_{x_m == x_l} 1
            CHECK (g_ll2kk2_histogram_integral_denormalized (point, interval_0_1, point, interval_0_1) == 1);
            CHECK (g_ll2kk2_histogram_integral_denormalized (point, interval_0_1, points, interval_0_1) == 1);
            CHECK (g_ll2kk2_histogram_integral_denormalized (points, interval_0_1, point, interval_0_1) == 1);
            CHECK (g_ll2kk2_histogram_integral_denormalized (points, interval_0_1, points, interval_0_1) == 3);
            // Simple cases for interval ]0,2]
            CHECK (g_ll2kk2_histogram_integral_denormalized (empty, interval_0_2, empty, interval_0_2) == 0);
            CHECK (g_ll2kk2_histogram_integral_denormalized (point, interval_0_2, empty, interval_0_2) == 0);
            CHECK (g_ll2kk2_histogram_integral_denormalized (point, interval_0_2, point, interval_0_2) == 2);
            // Results based on geometrical figures for more complex cases:
            //                                       ##     #
            // ##__ * (##__ + _##_ + __##) = ##__ * #### = ##__
            CHECK (g_ll2kk2_histogram_integral_denormalized (point, interval_0_2, points, interval_0_2) == 3);
            //                ##
            //                ##
            //  ##     ##     ##
            // #### * #### = ####
            CHECK (g_ll2kk2_histogram_integral_denormalized (points, interval_0_2, points, interval_0_2) == 10);
            //                         ##            ##     ##
            // (#___ + _#__ + __#_) * #### = ###_ * #### = ###_
            CHECK (g_ll2kk2_histogram_integral_denormalized (points, interval_0_1, points, interval_0_2) == 5);
            // Check that value is invariant by common interval shifting
            CHECK (g_ll2kk2_histogram_integral_denormalized (points, interval_0_2, points, interval_0_2) ==
                   g_ll2kk2_histogram_integral_denormalized (points, interval_2_4, points, interval_2_4));
    }
            TEST_CASE ("b_values") {
        const auto base = HistogramBase{2, 10}; // K=2, delta=10, intervals= ]0,10] ]10,20]
        const auto norm_factor = normalization_factor (base);
        const SortedVec<Point> points[] = {
            SortedVec<Point>::from_sorted ({0}),
            SortedVec<Point>::from_sorted ({0}),
        };
        fmt::print (stdout, "###############################\n{}\n", compute_b (make_span (points), base,
    None{}).inner);
      }*/
}

/******************************************************************************
 * Command line tests.
 */
TEST_SUITE("command_line") {
    TEST_CASE("usage") {
        // No actual checks, but useful to manually look at the formatting for different cases

        auto parser = CommandLineParser();

        SUBCASE("no arg, no opt") {}
        SUBCASE("args, no opt") {
            parser.positional("arg1", "Argument 1", [](string_view) {});
            parser.positional("arg_______2", "Argument 2", [](string_view) {});
        }
        SUBCASE("no args, opts") {
            parser.flag({"h", "help"}, "Shows help", []() {});
            parser.option({"f", "file"}, "file", "A file argument", [](string_view) {});
            parser.option2({"p"}, "first", "second", "A pair of values", [](string_view, string_view) {});
        }
        SUBCASE("args and opts") {
            parser.flag({"h", "help"}, "Shows help", []() {});
            parser.option({"f", "file"}, "file", "A file argument", [](string_view) {});
            parser.option2({"p"}, "first", "second", "A pair of values", [](string_view, string_view) {});
            parser.positional("arg1", "Argument 1", [](string_view) {});
            parser.positional("arg_______2", "Argument 2", [](string_view) {});
        }

        fmt::print(stdout, "#################################\n");
        parser.usage(stdout, "test");
    }

    TEST_CASE("construction_errors") {
        CommandLineParser parser;
        parser.flag({"h", "help"}, "Shows help", []() {});

        // No name
        CHECK_THROWS_AS(parser.flag({}, "blah", []() {}), CommandLineParser::Exception);
        // Empty name
        CHECK_THROWS_AS(parser.flag({""}, "blah", []() {}), CommandLineParser::Exception);
        // Name collision
        CHECK_THROWS_AS(parser.flag({"h"}, "blah", []() {}), CommandLineParser::Exception);
    }

    TEST_CASE("parsing") {
        {
            const char * argv_data[] = {"prog_name", "-f", "--f"};
            auto argv = CommandLineView(sizeof(argv_data) / sizeof(*argv_data), argv_data);

            // f as a flag should match both
            auto flag_parser = CommandLineParser();
            int flag_parser_f_seen = 0;
            flag_parser.flag({"f"}, "", [&]() { flag_parser_f_seen++; });
            flag_parser.parse(argv);
            CHECK(flag_parser_f_seen == 2);

            // f as value opt eats the second --f
            auto value_parser = CommandLineParser();
            value_parser.option({"f"}, "", "", [](string_view value) { CHECK(value == "--f"); });
            value_parser.parse(argv);

            // Fails because args look like opts that are not defined
            auto nothing_parser = CommandLineParser();
            CHECK_THROWS_AS(nothing_parser.parse(argv), CommandLineParser::Exception);

            // Success, no option declared
            auto arg_arg_parser = CommandLineParser();
            arg_arg_parser.positional("1", "", [](string_view v) { CHECK(v == "-f"); });
            arg_arg_parser.positional("2", "", [](string_view v) { CHECK(v == "--f"); });
            arg_arg_parser.parse(argv);

            // Fails, with options parsing enabled '-f' is unknown.
            auto arg_opt_parser = CommandLineParser();
            arg_opt_parser.flag({"z"}, "", []() {});
            arg_opt_parser.positional("1", "", [](string_view) {});
            arg_opt_parser.positional("2", "", [](string_view) {});
            CHECK_THROWS_AS(arg_opt_parser.parse(argv), CommandLineParser::Exception);
        }
        {
            const char * argv_data[] = {"prog_name", "1a", "a2"};
            auto argv = CommandLineView(sizeof(argv_data) / sizeof(*argv_data), argv_data);

            // One unexpected arg
            auto arg_parser = CommandLineParser();
            arg_parser.positional("1", "", [](string_view) {});
            CHECK_THROWS_AS(arg_parser.parse(argv), CommandLineParser::Exception);

            // Eats both args
            auto arg_arg_parser = CommandLineParser();
            arg_arg_parser.positional("1", "", [](string_view v) { CHECK(v == "1a"); });
            arg_arg_parser.positional("2", "", [](string_view v) { CHECK(v == "a2"); });
            arg_arg_parser.parse(argv);

            // Missing one arg
            auto arg_arg_arg_parser = CommandLineParser();
            arg_arg_arg_parser.positional("1", "", [](string_view) {});
            arg_arg_arg_parser.positional("2", "", [](string_view) {});
            arg_arg_arg_parser.positional("3", "", [](string_view) {});
            CHECK_THROWS_AS(arg_arg_arg_parser.parse(argv), CommandLineParser::Exception);
        }
        {
            // Value arg parsing
            const char * argv_data[] = {"prog_name", "-opt=value", "--opt", "value", "-opt", "value"};
            auto argv = CommandLineView(sizeof(argv_data) / sizeof(*argv_data), argv_data);

            auto parser = CommandLineParser();
            parser.option({"opt"}, "value", "desc", [](string_view v) { CHECK(v == "value"); });
            parser.parse(argv);
        }
        {
            // Value2 arg parsing
            const char * argv_data[] = {"prog_name", "-opt=value1", "value2", "--opt", "value1", "value2"};
            auto argv = CommandLineView(sizeof(argv_data) / sizeof(*argv_data), argv_data);

            auto parser = CommandLineParser();
            parser.option2({"opt"}, "value", "value2", "desc", [](string_view v1, string_view v2) {
                CHECK(v1 == "value1");
                CHECK(v2 == "value2");
            });
            parser.parse(argv);
        }
    }
}

/******************************************************************************
 * File reading.
 * For testing, we read this exact file.
 */
constexpr auto & this_file_name = __FILE__;
extern const int this_file_nb_lines; // Initialized at the bottom of this file.

TEST_SUITE("file_reading") {
    TEST_CASE("open_file") {
        auto f = open_file(this_file_name, "r");
        CHECK(f != nullptr);

        CHECK_THROWS(open_file("", "r"));
    }

    TEST_CASE("LineByLineReader") {
        auto f = open_file(this_file_name, "r");
        LineByLineReader reader(f.get());

        CHECK(reader.read_next_line() == true);
        CHECK(reader.current_line() == "// First line\n");
        CHECK(reader.current_line_number() == 0);

        CHECK(reader.read_next_line() == true);
        CHECK(reader.current_line() == "// Second line\n");
        CHECK(reader.current_line_number() == 1);

        while(reader.read_next_line()) {
            // Get all lines
        }
        CHECK(reader.current_line_number() == this_file_nb_lines - 1);
    }
}

/******************************************************************************
 * Test utilities.
 */
TEST_SUITE("utils") {
    TEST_CASE("split") {
        CHECK(split(',', "") == std::vector<string_view>{""});
        CHECK(split(',', ",") == std::vector<string_view>{"", ""});
        CHECK(split(',', ",,") == std::vector<string_view>{"", "", ""});
        CHECK(split(',', "a,b,c") == std::vector<string_view>{"a", "b", "c"});
        CHECK(split(',', "a,b") == std::vector<string_view>{"a", "b"});
        CHECK(split(',', "a,") == std::vector<string_view>{"a", ""});
        CHECK(split(',', ",b") == std::vector<string_view>{"", "b"});
        CHECK(split(',', " ,b ") == std::vector<string_view>{" ", "b "});
    }
    TEST_CASE("split_first_n") {
        auto r0 = split_first_n<1>(',', "");
        CHECK(r0.has_value == true);
        CHECK(r0.value[0] == "");
        auto r1 = split_first_n<2>(',', "");
        CHECK(r1.has_value == false);

        auto r2 = split_first_n<1>(',', "a,b");
        CHECK(r2.has_value == true);
        CHECK(r2.value[0] == "a");
        auto r3 = split_first_n<2>(',', "a,b");
        CHECK(r3.has_value == true);
        CHECK(r3.value[0] == "a");
        CHECK(r3.value[1] == "b");
        auto r4 = split_first_n<3>(',', "a,b");
        CHECK(r4.has_value == false);

        auto r5 = split_first_n<1>(',', "a,");
        CHECK(r5.has_value == true);
        CHECK(r5.value[0] == "a");
        auto r6 = split_first_n<2>(',', "a,");
        CHECK(r6.has_value == true);
        CHECK(r6.value[0] == "a");
        CHECK(r6.value[1] == "");
        auto r7 = split_first_n<3>(',', "a,");
        CHECK(r7.has_value == false);
    }
    TEST_CASE("trim_ws") {
        CHECK(trim_ws_left("") == "");
        CHECK(trim_ws_right("") == "");
        CHECK(trim_ws("") == "");

        CHECK(trim_ws_left(" \t\n") == "");
        CHECK(trim_ws_right(" \t\n") == "");
        CHECK(trim_ws(" \t\n") == "");

        CHECK(trim_ws_left(" a ") == "a ");
        CHECK(trim_ws_right(" a ") == " a");
        CHECK(trim_ws(" a ") == "a");
    }
    TEST_CASE("sorted_vec") {
        CHECK_THROWS(SortedVec<int>::from_sorted({1, 0}));
        CHECK_THROWS(SortedVec<int>::from_sorted({0, 0, 1}));
        const auto sorted = SortedVec<int>::from_unsorted({1, 0, 0});
        CHECK(sorted.size() == 2);
        CHECK(sorted[0] == 0);
        CHECK(sorted[1] == 1);
    }
    TEST_CASE("power_of_2") {
        CHECK(power_of_2(0) == 1);
        CHECK(power_of_2(1) == 2);
        CHECK(power_of_2(2) == 4);
    }
    TEST_CASE("floor_log2") {
        CHECK(floor_log2(1) == 0);
        CHECK(floor_log2(2) == 1);
        CHECK(floor_log2(3) == 1);
        CHECK(floor_log2(4) == 2);
        CHECK(floor_log2(5) == 2);
        CHECK(floor_log2(7) == 2);
        CHECK(floor_log2(8) == 3);
        CHECK(floor_log2(9) == 3);
    }
}

// __LINE__ counts from 1. Thus the line number of the last line is the number of lines of the file.
const int this_file_nb_lines = __LINE__; // MUST BE THE LAST LINE !
