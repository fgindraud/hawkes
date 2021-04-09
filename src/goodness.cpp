#include <cassert>
#include <memory>

#include "command_line.h"
#include "input.h"
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

    // Get current integral value up to current t
    double integral_value() const { return integral; }

    // Zero integral. Used to start integrating from 0 instead of -inf.
    void set_integral_value(double v) { integral = v; }

    // Incrementally integrate the sum from current t up to new_t.
    void advance_to_t(Point new_t) {
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
    }
};

/* Compute the set of lambda_hat_m for all x_m in N_m. */
std::vector<double> compute_lambda_hat_m_for_all_Nm(
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
    const auto normalization_factor = base.normalization_factor;
    // Prepare state
    auto lambda_hat_for_x_m = std::vector<double>();
    lambda_hat_for_x_m.reserve(points[m].size());
    auto integral_lk = Vector2d<IntegralSumPhiK>(nb_processes, base_size, [&](ProcessId l, FunctionBaseId k) {
        // Integrate up to 0. then reset accumulated value ; ready to start integrate from 0.
        auto integral = IntegralSumPhiK(points[l], base.histogram(k).interval);
        integral.advance_to_t(0.);
        integral.set_integral_value(0.);
        return integral;
    });
    // Compute incrementally
    for(const Point x_m : points[m]) {
        double sum_a_integrals_indicator = 0.;
        for(ProcessId l = 0; l < nb_processes; l += 1) {
            for(FunctionBaseId k = 0; k < base_size; k += 1) {
                auto & integral = integral_lk[l][k];
                integral.advance_to_t(x_m);
                sum_a_integrals_indicator += estimated_a.get_lk(m, l, k) * integral.integral_value();
            }
        }
        lambda_hat_for_x_m.push_back(sum_a_integrals_indicator * normalization_factor);
    }
    return lambda_hat_for_x_m;
}

/******************************************************************************
 * Program entry point.
 */
int main(int argc, char * argv[]) {

    const auto command_line = CommandLineView(argc, argv);
    auto parser = CommandLineParser();

    std::unique_ptr<HistogramBase> base;
    std::vector<ProcessFile> process_files;
    bool dump_region_info_option = false;

    parser.flag({"h", "help"}, "Display this help", [&]() {
        parser.usage(stderr, command_line.program_name());
        std::exit(EXIT_SUCCESS);
    });

    parser.option2(
        {"histogram"},
        "K",
        "delta",
        "Use an histogram base (k > 0, delta > 0)",
        [&base](string_view k_value, string_view delta_value) {
            const auto base_size = size_t(parse_strict_positive_int(k_value, "histogram K"));
            const auto delta = PointSpace(parse_strict_positive_double(delta_value, "histogram delta"));
            base = std::make_unique<HistogramBase>(base_size, delta);
        });

    // File parsing : generate a list of (file, options), read later.
    parser.option(
        {"f", "file-forward"}, "filename", "Add process regions from file", [&process_files](string_view filename) {
            process_files.emplace_back(ProcessFile{filename, ProcessDirection::Forward});
        });
    parser.option(
        {"b", "file-backward"},
        "filename",
        "Add process regions (reversed) from file",
        [&process_files](string_view filename) {
            process_files.emplace_back(ProcessFile{filename, ProcessDirection::Backward});
        });

    parser.flag(
        {"dump-region-info"}, "Stop after parsing and print region/process point counts", [&dump_region_info_option]() {
            dump_region_info_option = true;
        });

    try {
        // Parse command line arguments. All actions declared to the parser will be called here.
        parser.parse(command_line);

        // Print region point counts and stop if requested
        if(dump_region_info_option) {
            print_region_info(process_files);
            return EXIT_SUCCESS;
        }

        if(base == nullptr) {
            throw std::runtime_error("Function base is not defined");
        }

        // Read input files
        const auto data_points = read_process_files(process_files);
        const auto points = extract_point_lists(data_points);

        // TODO parsing from file
        auto estimated_a = Matrix_M_MK1(points.nb_processes(), base->base_size);

    } catch(const CommandLineParser::Exception & exc) {
        fmt::print(stderr, "Error: {}. Use --help for a list of options.\n", exc.what());
        return EXIT_FAILURE;
    } catch(const std::exception & exc) {
        fmt::print(stderr, "Error: {}\n", exc.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}