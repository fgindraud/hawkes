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
struct SupportedKernelConfig {
    string_view name;

    // Function used to build config from input
    std::unique_ptr<KernelConfig> (*build_function)(
        const DataByProcessRegion<SortedVec<DataPoint>> & data_points,
        const Optional<std::vector<PointSpace>> & override_homogeneous_kernel_widths);
};
static const SupportedKernelConfig supported_kernel_configs[] = {
    {
        "none",
        [](const DataByProcessRegion<SortedVec<DataPoint>> &, const Optional<std::vector<PointSpace>> &)
            -> std::unique_ptr<KernelConfig> { return std::make_unique<NoKernel>(); },
    },
    {
        "homogeneous_interval",
        [](const DataByProcessRegion<SortedVec<DataPoint>> & data_points,
           const Optional<std::vector<PointSpace>> & override_homogeneous_kernel_widths)
            -> std::unique_ptr<KernelConfig> {
            return std::make_unique<HomogeneousKernels<IntervalKernel>>(
                determine_homogeneous_kernels<IntervalKernel>(data_points, override_homogeneous_kernel_widths));
        },
    },
    {
        "heterogeneous_interval",
        [](const DataByProcessRegion<SortedVec<DataPoint>> & data_points,
           const Optional<std::vector<PointSpace>> &) -> std::unique_ptr<KernelConfig> {
            return std::make_unique<HeterogeneousKernels<IntervalKernel>>(
                extract_heterogeneous_kernels<IntervalKernel>(data_points));
        },
    },
};

static std::string kernel_config_joined_names() {
    std::string joined;
    for(const SupportedKernelConfig & config : supported_kernel_configs) {
        if(!joined.empty()) {
            joined.push_back('|');
        }
        joined.append(config.name.begin(), config.name.end());
    }
    return joined;
}

static std::unique_ptr<KernelConfig> build_kernel_config(
    string_view name,
    const DataByProcessRegion<SortedVec<DataPoint>> & data_points,
    const Optional<std::vector<PointSpace>> & override_homogeneous_kernel_widths) {

    for(const SupportedKernelConfig & config : supported_kernel_configs) {
        if(config.name == name) {
            return config.build_function(data_points, override_homogeneous_kernel_widths);
        }
    }
    throw std::runtime_error(fmt::format("Unknown kernel configuration name: {}", name));
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
        kernel_config_joined_names(),
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
        const auto data_points = read_process_files(process_files);

        // Post processing: determine kernel setup, generate sorted points lists
        const auto post_processing_start = instant();
        const auto points = extract_point_lists(data_points);
        const auto kernel_config =
            build_kernel_config(kernel_config_option, data_points, override_homogeneous_kernel_widths);
        const auto post_processing_end = instant();
        fmt::print(
            stderr, "Post processing done: time = {}\n", duration_string(post_processing_end - post_processing_start));

        // Compute base/kernel specific values: B, G, B_hat
        // Ths visit() call generates an if/else if/.../else block for all combinations of base and kernel setup.
        // For each case, an overload of compute_intermediate_values indicated by its argument types does the
        // computation.
        const auto compute_b_g_start = instant();
        const CommonIntermediateValues intermediate_values{}; // FIXME
        const auto compute_b_g_end = instant();
        fmt::print(
            stderr,
            "Computing B and G matrice done: time = {}\n",
            duration_string(compute_b_g_end - compute_b_g_start));

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
