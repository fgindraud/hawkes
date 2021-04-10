// Main hawkes inferrence tool
#include <memory>
#include <string>
#include <utility>

#include <cassert>
#include <random>

#include "command_line.h"
#include "computations.h"
#include "input.h"
#include "lassoshooting.h"
#include "utils.h"

/******************************************************************************
 * Kernel configuration selection and construction.
 */
static const string_view kernel_config_option_values[] = {
    "none",
    "homogeneous_interval",
    "heterogeneous_interval",
};

static std::unique_ptr<KernelConfig> build_kernel_config(
    string_view option_value,
    const DataByProcessRegion<SortedVec<DataPoint>> & data_points,
    const Optional<std::vector<PointSpace>> & override_homogeneous_kernel_widths) {

    if(option_value == "none") {
        return std::make_unique<NoKernel>();
    } else if(option_value == "homogeneous_interval") {
        return std::make_unique<HomogeneousKernels<IntervalKernel>>(
            determine_homogeneous_kernels<IntervalKernel>(data_points, override_homogeneous_kernel_widths));
    } else if(option_value == "heterogeneous_interval") {
        return std::make_unique<HeterogeneousKernels<IntervalKernel>>(
            extract_heterogeneous_kernels<IntervalKernel>(data_points));
    } else {
        throw std::runtime_error(fmt::format("Unknown kernel configuration name: {}", option_value));
    }
}

/******************************************************************************
 * Program entry point.
 */
int main(int argc, char * argv[]) {
    const Eigen::IOFormat eigen_format(Eigen::FullPrecision, 0, "\t"); // Matrix printing format, used later

    bool verbose = false;
    double gamma = 1.;
    double lambda = 1.;

    std::unique_ptr<Base> base;
    string_view kernel_config_option = "none";
    Optional<std::vector<PointSpace>> override_homogeneous_kernel_widths;

    std::vector<ProcessFile> process_files;

    bool skip_reestimation = false;
    bool dump_region_info_option = false;
    bool dump_intermediate_values = false;

    // Command line parsing setup
    const auto command_line = CommandLineView(argc, argv);
    auto parser = CommandLineParser();

    // Common
    parser.flag({"h", "help"}, "Display this help", [&]() {
        parser.usage(stderr, command_line.program_name());
        std::exit(EXIT_SUCCESS);
    });
    parser.flag({"v", "verbose"}, "Enable verbose output", [&]() { verbose = true; });

    // Hyper-parameters
    parser.option({"g", "gamma"}, "value", "Set gamma value (double, positive)", [&gamma](string_view value) {
        gamma = parse_strict_positive_double(value, "gamma");
    });
    parser.option(
        {"lambda"}, "value", "Set global penalty multiplier (double, positive)", [&lambda](string_view value) {
            lambda = parse_strict_positive_double(value, "lambda");
        });

    // Base
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
    parser.option2(
        {"haar"},
        "nb_scales",
        "delta",
        "Use Haar square wavelets (nb_scales > 0, delta > 0)",
        [&base](string_view k_value, string_view delta_value) {
            const auto nb_scales = size_t(parse_strict_positive_int(k_value, "haar nb_scales"));
            if(!(nb_scales <= HaarBase::max_nb_scales)) {
                throw std::runtime_error(fmt::format("Haar nb_scales is limited to {}", HaarBase::max_nb_scales));
            }
            const auto delta = PointSpace(parse_strict_positive_double(delta_value, "haar delta"));
            if(delta <= double(power_of_2(nb_scales))) {
                fmt::print(
                    stderr,
                    "Warning: Haar wavelet base: smallest scale is less than 1.\n"
                    "This may be an error for integer data coordinates.\n");
            }
            base = std::make_unique<HaarBase>(nb_scales, delta);
        });

    // Kernel setup
    parser.option(
        {"k", "kernel"},
        fmt::format("{}", fmt::join(kernel_config_option_values, "|")),
        "Kernel configuration (default=none)",
        [&kernel_config_option](string_view value) { kernel_config_option = value; });
    parser.option(
        {"homogeneous-kernel-widths"},
        "w0[:w1:w2:...]",
        "Override kernel widths (default=median)",
        [&override_homogeneous_kernel_widths](string_view values) {
            override_homogeneous_kernel_widths = map_to_vector(split(':', values), [](string_view value) {
                return PointSpace(parse_strict_positive_double(value, "kernel width"));
            });
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

    // Debugging options
    parser.flag({"skip-reestimation"}, "Do not perform re-estimation after Lasso", [&skip_reestimation]() {
        skip_reestimation = true;
    });
    parser.flag(
        {"dump-region-info"}, "Stop after parsing and print region/process point counts", [&dump_region_info_option]() {
            dump_region_info_option = true;
        });
    parser.flag({"print-supported-computations"}, "Print the list of supported computations and stops", []() {
        print_supported_computation_cases();
        std::exit(EXIT_SUCCESS);
    });
    parser.flag({"dump-intermediate-values"}, "Print the B,G,D matrices used in Lasso", [&dump_intermediate_values]() {
        dump_intermediate_values = true;
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
        const ProcessFilesContent process_files_content = read_process_files(process_files);

        // Post processing: determine kernel setup, generate sorted points lists
        const auto post_processing_start = instant();
        const DataByProcessRegion<SortedVec<Point>> points = extract_point_lists(process_files_content.points);
        const std::unique_ptr<KernelConfig> kernel_config =
            build_kernel_config(kernel_config_option, process_files_content.points, override_homogeneous_kernel_widths);
        const auto post_processing_end = instant();
        fmt::print(
            stderr, "Post processing done: time = {}\n", duration_string(post_processing_end - post_processing_start));

        // Compute base/kernel specific values: B, G, V_hat, B_hat
        const auto compute_intermediates_start = instant();
        std::vector<IntermediateValues> intermediate_values =
            compute_intermediate_values(points, *base, *kernel_config);
        const auto compute_intermediates_end = instant();
        fmt::print(
            stderr,
            "Computing B and G matrice done: time = {}\n",
            duration_string(compute_intermediates_end - compute_intermediates_start));

        // Perform lassoshooting and final re-estimation
        const auto lasso_start = instant();
        const LassoParameters lasso_parameters = compute_lasso_parameters(intermediate_values, gamma);
        if(dump_intermediate_values) {
            fmt::print(
                "# B matrix (rows = {{0}} U {{(l,k)}}, cols = {{m}})\n{}\n",
                lasso_parameters.sum_of_b.inner.format(eigen_format));
            fmt::print(
                "# G matrix (rows & cols = {{0}} U {{(l,k)}})\n{}\n",
                lasso_parameters.sum_of_g.inner.format(eigen_format));
            fmt::print(
                "# V_hat matrix (rows = {{0}} U {{(l,k)}}, cols = {{m}})\n{}\n",
                lasso_parameters.sum_of_v_hat.inner.format(eigen_format));
            fmt::print(
                "# B_hat matrix (rows = {{0}} U {{(l,k)}}, cols = {{m}})\n{}\n",
                lasso_parameters.sum_of_b_hat.inner.format(eigen_format));
            fmt::print(
                "# D matrix (rows = {{0}} U {{(l,k)}}, cols = {{m}})\n{}\n",
                lasso_parameters.d.inner.format(eigen_format));
            fmt::print("# Estimated A\n"); // Add a separator between D and A (when printed later).
        }
        Matrix_M_MK1 estimated_a = compute_estimated_a_with_lasso(lasso_parameters, lambda);
        if(!skip_reestimation) {
            estimated_a = compute_reestimated_a(lasso_parameters, estimated_a);
        }
        const auto lasso_end = instant();
        fmt::print(stderr, "Lassoshooting done: time = {}\n", duration_string(lasso_end - lasso_start));

        // Print results
        if(verbose) {
            // Header
            fmt::print("# Processes = {{\n");
            for(ProcessId m = 0; m < process_files.size(); ++m) {
                const auto & p = process_files[m];
                const string_view suffix = p.direction == ProcessDirection::Backward ? " (backward)" : "";
                fmt::print("#  [{}] {}{}\n", m, p.filename, suffix);
            }
            fmt::print("# }}\n");

            base->write_verbose_description(stdout);
            kernel_config->write_verbose_description(stdout);

            fmt::print(
                "# gamma = {}\n"
                "# lambda = {}\n"
                "# re-estimation after Lasso = {}\n",
                gamma,
                lambda,
                !skip_reestimation);

            fmt::print("# Rows = {{0}} U {{(l,k)}} (order = 0,(0,0),..,(0,K-1),(1,0),..,(1,K-1),...,(M-1,K-1))\n");
            fmt::print("# Columns = {{m}}\n");
        }
        fmt::print("{}\n", estimated_a.inner.format(eigen_format));

    } catch(const CommandLineParser::Exception & exc) {
        fmt::print(stderr, "Error: {}. Use --help for a list of options.\n", exc.what());
        return EXIT_FAILURE;
    } catch(const std::exception & exc) {
        fmt::print(stderr, "Error: {}\n", exc.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
