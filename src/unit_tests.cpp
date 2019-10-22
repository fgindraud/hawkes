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

namespace doctest {
template <typename T> struct StringMaker<std::vector<T>> {
    static String convert(const std::vector<T> & v) {
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
};
template <> struct StringMaker<string_view> {
    static String convert(string_view sv) {
        using doctest::toString;
        return toString("\"") + toString(to_string(sv)) + toString("\"");
    }
};
} // namespace doctest

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
TEST_SUITE("shape") {
    using namespace shape;
    TEST_CASE("interval") {
        const auto interval = IntervalIndicator::with_half_width(1); // [-1, 1]
        const auto nzd = interval.non_zero_domain();
        CHECK(nzd == ClosedInterval{-1, 1});
        CHECK(contains(nzd, 0));
        CHECK(contains(nzd, 1));
        CHECK(!contains(nzd, 2));
        CHECK(interval(-2) == 0);
        CHECK(interval(-1) == 1);
        CHECK(interval(0) == 1);
        CHECK(interval(1) == 1);
        CHECK(interval(2) == 0);
    }
    TEST_CASE("combinators") {
        const auto interval = IntervalIndicator::with_half_width(1); // [-1, 1]
        const auto scaled_2 = scaled(2, interval);
        CHECK(scaled_2(0) == 2);
        CHECK(scaled_2(1) == 2);
        CHECK(scaled_2(2) == 0);
        CHECK(scaled_2.non_zero_domain() == interval.non_zero_domain());
        const auto shifted_forward = shifted(1, interval);
        CHECK(shifted_forward(-2) == 0);
        CHECK(shifted_forward(-1) == 0);
        CHECK(shifted_forward(0) == 1);
        CHECK(shifted_forward(1) == 1);
        CHECK(shifted_forward(2) == 1);
        CHECK(shifted_forward(3) == 0);
        CHECK(shifted_forward.non_zero_domain() == ClosedInterval{0, 2});
        const auto rev = reversed(shifted_forward); // Use the non symmetric shifted_forward shape
        CHECK(rev(-3) == 0);
        CHECK(rev(-2) == 1);
        CHECK(rev(-1) == 1);
        CHECK(rev(0) == 1);
        CHECK(rev(1) == 0);
        CHECK(rev(2) == 0);
        CHECK(rev.non_zero_domain() == ClosedInterval{-2, 0});
    }
    TEST_CASE("triangles") {
        const auto pos_tri = PositiveTriangle(2);
        CHECK(pos_tri(-1) == 0);
        CHECK(pos_tri(0) == 0);
        CHECK(pos_tri(1) == 1);
        CHECK(pos_tri(2) == 2);
        CHECK(pos_tri(3) == 0);
        const auto neg_tri = NegativeTriangle(2);
        CHECK(neg_tri(-3) == 0);
        CHECK(neg_tri(-2) == 2);
        CHECK(neg_tri(-1) == 1);
        CHECK(neg_tri(0) == 0);
        CHECK(neg_tri(1) == 0);
    }
    TEST_CASE("trapezoid") {
        const auto interval = IntervalIndicator::with_half_width(1); // [-1, 1]
        const auto degenerate_trapezoid = convolution(interval, interval);
        // The trapezoid is a simple triangle centered on 0 (no central section).
        const auto degenerate_trapezoid_nzd = degenerate_trapezoid.non_zero_domain();
        CHECK(degenerate_trapezoid_nzd.left == -degenerate_trapezoid_nzd.right);
        CHECK(degenerate_trapezoid_nzd.right == 2 * interval.half_width);
        CHECK(degenerate_trapezoid.half_base == 0);
        CHECK(degenerate_trapezoid.height == 2); // Max height == integral_x interval(x) == 1*width == 2
        // Graph
        CHECK(degenerate_trapezoid(-3) == 0);
        CHECK(degenerate_trapezoid(-2) == 0);
        CHECK(degenerate_trapezoid(-1) == 1);
        CHECK(degenerate_trapezoid(0) == 2);
        CHECK(degenerate_trapezoid(1) == 1);
        CHECK(degenerate_trapezoid(2) == 0);
        CHECK(degenerate_trapezoid(3) == 0);
        // Non degenerate trapezoid
        const auto trapezoid = convolution(interval, IntervalIndicator::with_half_width(2));
        CHECK(trapezoid(-4) == 0);
        CHECK(trapezoid(-3) == 0);
        CHECK(trapezoid(-2) == 1);
        CHECK(trapezoid(-1) == 2);
        CHECK(trapezoid(0) == 2);
        CHECK(trapezoid(1) == 2);
        CHECK(trapezoid(2) == 1);
        CHECK(trapezoid(3) == 0);
        CHECK(trapezoid(4) == 0);
        // Components
        const auto left_tri = component(trapezoid, Trapezoid::LeftTriangle{});
        CHECK(left_tri(-3) == 0);
        CHECK(left_tri(-2) == 1);
        CHECK(left_tri(-1) == 2);
        CHECK(left_tri(0) == 0);
        const auto central_block = component(trapezoid, Trapezoid::CentralBlock{});
        CHECK(central_block(-2) == 0);
        CHECK(central_block(-1) == 2);
        CHECK(central_block(0) == 2);
        CHECK(central_block(1) == 2);
        CHECK(central_block(2) == 0);
        const auto right_tri = component(trapezoid, Trapezoid::RightTriangle{});
        CHECK(right_tri(0) == 0);
        CHECK(right_tri(1) == 2);
        CHECK(right_tri(2) == 1);
        CHECK(right_tri(3) == 0);
    }
    TEST_CASE("ConvolutionIntervalPositiveTriangle") {
        {
            // Case l <= c
            const auto interval = IntervalIndicator::with_width(2);
            const auto tri = PositiveTriangle(3);
            const auto conv = convolution(interval, tri);
            CHECK(conv.non_zero_domain() == ClosedInterval{-1, 3 + 1});
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
            // Case l >= c
            const auto interval = IntervalIndicator::with_width(2);
            const auto tri = PositiveTriangle(1);
            const auto conv = convolution(interval, tri);
            CHECK(conv.non_zero_domain() == ClosedInterval{-1, 1 + 1});
            CHECK(conv(-2) == 0);
            CHECK(conv(-1) == 0); // f(-l/2) == 0
            CHECK(conv(0) > conv(-1));
            CHECK(conv(0) == 1. * 1. / 2.); // f(c-l/2) = c^2/2
            CHECK(conv(1) == 1. * 1. / 2.); // f(l/2) = c^2/2
            CHECK(conv(2) < conv(1));
            CHECK(conv(2) == 0);
            CHECK(conv(3) == 0);
        }
    }
    TEST_CASE("ConvolutionPositiveTrianglePositiveTriangle") {
        // 2 symmetrical cases: use a=3,b=4 so that the max of the shape is at 5.
        const auto a_tri = PositiveTriangle(3);
        const auto b_tri = PositiveTriangle(4);
        const auto conv = convolution(a_tri, b_tri);
        CHECK(conv.non_zero_domain() == ClosedInterval{0, 3 + 4});
        CHECK(conv(-1) == 0);
        CHECK(conv(0) == 0);
        CHECK(conv(1) > conv(0));
        CHECK(conv(2) > conv(1));
        CHECK(conv(3) > conv(2));
        CHECK(conv(3) == 3. * 3. * 3. / 6.);
        CHECK(conv(4) > conv(3));
        CHECK(conv(4) == 3. * 3. * (3. * 4. - 2. * 3.) / 6.);
        CHECK(conv(5) > conv(4));
        CHECK(conv(6) < conv(5));
        CHECK(conv(7) < conv(6));
        CHECK(conv(7) == 0);
        CHECK(conv(8) == 0);
    }
    TEST_CASE("ConvolutionNegativeTrianglePositiveTriangle") {
        // 2 cases with results time reverted from the other.
        const auto neg_tri = NegativeTriangle(2);
        const auto pos_tri = PositiveTriangle(3);
        const auto conv = convolution(neg_tri, pos_tri);
        CHECK(conv.non_zero_domain() == ClosedInterval{-2, 3});
        CHECK(conv(-3) == 0);
        CHECK(conv(-2) == 0);
        CHECK(conv(-1) > conv(-2));
        CHECK(conv(0) > conv(-1));
        CHECK(conv(0) == 2. * 2. * 2. / 3.);
        CHECK(conv(1) > conv(0));
        CHECK(conv(1) == 2. * 2. * (3. * 3. - 2.) / 6.);
        CHECK(conv(2) < conv(1));
        CHECK(conv(3) < conv(2));
        CHECK(conv(3) == 0);
        CHECK(conv(4) == 0);
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
    TEST_CASE("sum_of_point_differences") {
        // Use interval [-1, 1] as this is a simple function to check
        const auto interval = shape::IntervalIndicator::with_half_width(1);
        const auto empty = SortedVec<Point>::from_sorted({});
        const auto zero = SortedVec<Point>::from_sorted({0});
        // Should be zero due to emptyset
        CHECK(sum_of_point_differences(empty, empty, interval) == 0);
        CHECK(sum_of_point_differences(empty, zero, interval) == 0);
        CHECK(sum_of_point_differences(zero, empty, interval) == 0);
        // Only one same point in both sets
        CHECK(sum_of_point_differences(zero, zero, interval) == 1);
        // Single point with all points near zero
        const auto all_near_zero = SortedVec<Point>::from_sorted({-4, -3, -2, -1, 0, 1, 2, 3, 4});
        CHECK(sum_of_point_differences(zero, all_near_zero, interval) == 3);
        CHECK(sum_of_point_differences(all_near_zero, zero, interval) == 3);
        // Multiple points with all points near zero
        const auto some_points = SortedVec<Point>::from_sorted({-3, 0, 3});
        CHECK(sum_of_point_differences(some_points, all_near_zero, interval) == 9);
        CHECK(sum_of_point_differences(all_near_zero, some_points, interval) == 9);
        CHECK(sum_of_point_differences(some_points, some_points, interval) == 3);
    }
    TEST_CASE("sup_of_sum_of_differences_to_points: IntervalIndicator") {
        const auto interval = shape::IntervalIndicator::with_half_width(1);
        SUBCASE("no points") {
            const auto vec = SortedVec<Point>::from_sorted({});
            CHECK(sup_of_sum_of_differences_to_points(vec, interval) == 0);
        }
        SUBCASE("non overlapping") {
            const auto vec = SortedVec<Point>::from_sorted({0, 3});
            CHECK(sup_of_sum_of_differences_to_points(vec, interval) == 1);
        }
        SUBCASE("overlapping: inner") {
            const auto vec = SortedVec<Point>::from_sorted({0, 1, 4, 5});
            CHECK(sup_of_sum_of_differences_to_points(vec, interval) == 2);
        }
        SUBCASE("overlapping: edge") {
            const auto vec = SortedVec<Point>::from_sorted({0, 2, 4});
            CHECK(sup_of_sum_of_differences_to_points(vec, interval) == 2);
        }
        SUBCASE("overlapping: edge and inner") {
            const auto vec = SortedVec<Point>::from_sorted({0, 1, 2, 3, 4});
            CHECK(sup_of_sum_of_differences_to_points(vec, interval) == 3);
        }
    }
    TEST_CASE("sup_of_sum_of_differences_to_points: Interval<Open, Closed>") {
        const auto interval = Interval<Bound::Open, Bound::Closed>{0, 3}; // ]0,3]
        SUBCASE("no points") {
            const auto vec = SortedVec<Point>::from_sorted({});
            CHECK(sup_of_sum_of_differences_to_points(vec, interval) == 0);
        }
        SUBCASE("non overlapping") {
            const auto vec = SortedVec<Point>::from_sorted({0, 3, 6});
            CHECK(sup_of_sum_of_differences_to_points(vec, interval) == 1);
        }
        SUBCASE("overlapping") {
            const auto vec = SortedVec<Point>::from_sorted({0, 2});
            CHECK(sup_of_sum_of_differences_to_points(vec, interval) == 2);
        }
        SUBCASE("multiple overlapping") {
            const auto vec = SortedVec<Point>::from_sorted({0, 1, 2});
            CHECK(sup_of_sum_of_differences_to_points(vec, interval) == 3);
        }
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
