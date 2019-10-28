#include <algorithm>
#include <chrono>
#include <string>
#include <unordered_set>

#include <cassert>
#include <random>

#include "command_line.h"
#include "computations.h"
#include "input.h"
#include "lassoshooting.h"
#include "utils.h"

/******************************************************************************
 * Time measurement primitives.
 */
template <typename Rep, typename Period>
static std::string duration_string(std::chrono::duration<Rep, Period> duration) {
    namespace chrono = std::chrono;
    using chrono::duration_cast;
    const auto hours = duration_cast<chrono::hours>(duration).count();
    if(hours > 10) {
        return fmt::format("{}h", hours);
    }
    const auto minutes = duration_cast<chrono::minutes>(duration).count();
    if(minutes > 10) {
        return fmt::format("{}m", minutes);
    }
    const auto seconds = duration_cast<chrono::seconds>(duration).count();
    if(seconds > 10) {
        return fmt::format("{}s", seconds);
    }
    const auto milliseconds = duration_cast<chrono::milliseconds>(duration).count();
    if(milliseconds > 10) {
        return fmt::format("{}ms", milliseconds);
    }
    const auto microseconds = duration_cast<chrono::microseconds>(duration).count();
    if(microseconds > 10) {
        return fmt::format("{}us", microseconds);
    }
    const auto nanoseconds = duration_cast<chrono::nanoseconds>(duration).count();
    return fmt::format("{}ns", nanoseconds);
}

static std::chrono::high_resolution_clock::time_point instant() {
    return std::chrono::high_resolution_clock::now();
}

/******************************************************************************
 * Program entry point.
 */

enum class KernelConfig {
    None,          // No Kernel, "ponctual" mode
    Homogeneous,   // One kernel per process, chosen as median of BED interval widths by default
    Heterogeneous, // One kernel per point, chosen as width of the BED interval from which the point was generated
};
enum class KernelType {
    Interval,          // L2-normalized indicator function centered on 0
    IntervalRightHalf, // L2-normalized indicator function on [0, width/2]
};

using BaseType = variant<HistogramBase, HaarBase>;
using BaseOption = variant<None, HistogramBase, HaarBase>; // Option type, maybe undefined (None).

enum class ProcessDirection {
    Forward,  // Keep points coordinates
    Backward, // x -> -x for all points coordinates
};

struct ProcessFile {
    string_view filename;
    ProcessDirection direction;
};

static BedFileRegions read_regions_from(string_view filename) {
    try {
        const auto start = instant();
        auto file = open_file(filename, "r");
        auto regions = BedFileRegions::read_from(file.get());
        const auto end = instant();
        fmt::print(
            stderr,
            "Process {} loaded: regions = {} ; time = {}\n",
            filename,
            regions.nb_regions(),
            duration_string(end - start));
        return regions;
    } catch(const std::runtime_error & e) {
        throw std::runtime_error(fmt::format("Reading process data from {}: {}", filename, e.what()));
    }
}

static std::vector<std::string> union_of_all_region_names(const std::vector<BedFileRegions> & bed_files_content) {
    // Generate the list of all discovered region names (some may be missing from some files if empty).
    // The list is returned as a vector to guarantee the same order of regions for all processes.
    std::unordered_set<std::string> region_name_set;
    for(const auto & bed_file : bed_files_content) {
        for(const auto & region : bed_file.region_by_name) {
            region_name_set.emplace(region.first);
        }
    }
    return std::vector<std::string>(region_name_set.begin(), region_name_set.end());
}

static void print_region_info(const std::vector<ProcessFile> & files) {
    auto bed_files_content = map_to_vector(files, [](const ProcessFile & f) { return read_regions_from(f.filename); });
    // Header
    fmt::print("region_name");
    for(const auto & file : files) {
        fmt::print("\t{}", file.filename);
    }
    fmt::print("\n");
    // Table
    for(const auto & region_name : union_of_all_region_names(bed_files_content)) {
        fmt::print("{}", region_name);
        for(const auto & content : bed_files_content) {
            auto region_entry = content.region_by_name.find(region_name);
            if(region_entry != content.region_by_name.end()) {
                const auto & region = region_entry->second;
                fmt::print("\t{}", region.unsorted_points.size());
            } else {
                fmt::print("\tmissing");
            }
        }
        fmt::print("\n");
    }
}

static DataByProcessRegion<SortedVec<DataPoint>> read_process_files(const std::vector<ProcessFile> & files) {
    if(files.empty()) {
        throw std::runtime_error("read_process_files: process list is empty");
    }
    auto bed_files_content = map_to_vector(files, [](const ProcessFile & f) { return read_regions_from(f.filename); });
    const std::vector<std::string> all_region_names = union_of_all_region_names(bed_files_content);

    // Fill a 2D matrix of lists of PointInterval with contents from bed files.
    // Missing regions in some processes will lead to empty lists.
    // Number of missing regions is tracked per process for information (if high, may indicate naming mismatch).
    const auto nb_processes = files.size();
    const auto nb_regions = all_region_names.size();
    DataByProcessRegion<SortedVec<DataPoint>> points(nb_processes, nb_regions);

    fmt::print(stderr, "Missing region ratio (per process):");
    for(ProcessId m = 0; m < nb_processes; m++) {
        const auto & file = files[m];
        auto & content = bed_files_content[m];
        std::size_t number_missing = 0;
        for(RegionId r = 0; r < nb_regions; ++r) {
            auto region_entry = content.region_by_name.find(all_region_names[r]);
            if(region_entry != content.region_by_name.end()) {
                auto & unsorted_points = region_entry->second.unsorted_points;
                // Apply reversing if requested before sorting them in increasing order
                if(file.direction == ProcessDirection::Backward) {
                    for(auto & data_point : unsorted_points) {
                        data_point.center = -data_point.center;
                    }
                }
                points.data(m, r) = SortedVec<DataPoint>::from_unsorted(std::move(unsorted_points));
            } else {
                // Do nothing, points.data(m,r) has been initialized to an empty list of points
                number_missing += 1;
            }
        }
        fmt::print(stderr, "  {}%", double(number_missing) / double(nb_regions) * 100.);
    }
    fmt::print(stderr, "\n");
    return points;
}

static DataByProcessRegion<SortedVec<Point>> extract_point_lists(
    const DataByProcessRegion<SortedVec<DataPoint>> & data_points) {
    const auto nb_processes = data_points.nb_processes();
    const auto nb_regions = data_points.nb_regions();
    DataByProcessRegion<SortedVec<Point>> points(nb_processes, nb_regions);
    for(ProcessId m = 0; m < nb_processes; ++m) {
        for(RegionId r = 0; r < nb_regions; ++r) {
            const auto & region_data_points = data_points.data(m, r);
            const auto region_size = region_data_points.size();
            std::vector<Point> region_points(region_size);
            for(size_t i = 0; i < region_size; ++i) {
                region_points[i] = region_data_points[i].center;
            }
            points.data(m, r) = SortedVec<Point>::from_sorted(std::move(region_points));
        }
    }
    return points;
}

// Generate kernels and maximum width kernels from data_points
template <typename WidthToKernelFunc> static auto get_heterogeneous_kernels(
    const DataByProcessRegion<SortedVec<DataPoint>> & data_points, WidthToKernelFunc width_to_kernel) {
    using KernelT = decltype(width_to_kernel(PointSpace()));
    const auto nb_processes = data_points.nb_processes();
    const auto nb_regions = data_points.nb_regions();

    DataByProcessRegion<std::vector<KernelT>> kernels(nb_processes, nb_regions);
    std::vector<KernelT> maximum_width_kernels;

    for(ProcessId m = 0; m < nb_processes; ++m) {
        PointSpace max_width = 0.;
        for(RegionId r = 0; r < nb_regions; ++r) {
            const auto & region_data_points = data_points.data(m, r);
            std::vector<KernelT> region_kernels;
            region_kernels.reserve(region_data_points.size());
            for(const auto & data_point : region_data_points) {
                const auto width = fix_zero_width(data_point.width);
                region_kernels.emplace_back(width_to_kernel(width));
                max_width = std::max(max_width, width);
            }
            kernels.data(m, r) = std::move(region_kernels);
        }
        maximum_width_kernels.emplace_back(width_to_kernel(max_width));
    }
    return HeterogeneousKernels<KernelT>{std::move(kernels), std::move(maximum_width_kernels)};
}

static std::vector<PointSpace> median_interval_widths(const DataByProcessRegion<SortedVec<DataPoint>> & data_points) {
    const auto compute_median_width = [&data_points](ProcessId m) {
        const auto sum_of_region_sizes = [&data_points, m]() {
            size_t sum = 0;
            for(RegionId r = 0; r < data_points.nb_regions(); ++r) {
                sum += data_points.data(m, r).size();
            }
            return sum;
        }();
        // Degenerate case
        if(sum_of_region_sizes == 0) {
            return PointSpace(0);
        }
        // Build a vector containing the unsorted widths of all intervals from all regions
        std::vector<PointSpace> all_widths;
        all_widths.reserve(sum_of_region_sizes);
        for(RegionId r = 0; r < data_points.nb_regions(); ++r) {
            const auto & region_data_points = data_points.data(m, r);
            for(const auto & data_point : region_data_points) {
                all_widths.emplace_back(data_point.width);
            }
        }
        assert(all_widths.size() == sum_of_region_sizes);
        // Compute the median in the naive way. In C++17 std::nth_element would be a better O(n) solution.
        std::sort(all_widths.begin(), all_widths.end());
        assert(sum_of_region_sizes > 0);
        if(sum_of_region_sizes % 2 == 1) {
            const auto mid_point_index = sum_of_region_sizes / 2;
            return all_widths[mid_point_index];
        } else {
            const auto above_mid_point_index = sum_of_region_sizes / 2;
            assert(above_mid_point_index > 0);
            return (all_widths[above_mid_point_index - 1] + all_widths[above_mid_point_index]) / 2.;
        }
    };

    std::vector<PointSpace> medians(data_points.nb_processes());
    for(ProcessId m = 0; m < data_points.nb_processes(); ++m) {
        medians[m] = fix_zero_width(compute_median_width(m));
    }
    return medians;
}

static variant<None, std::vector<IntervalKernel>, HeterogeneousKernels<IntervalKernel>> determine_kernel_setup(
    const DataByProcessRegion<SortedVec<DataPoint>> & data_points,
    KernelConfig config,
    KernelType kernel_type,
    Optional<std::vector<PointSpace>> & override_homogeneous_kernel_widths) {
    // Utils
    auto get_homogeneous_kernel_widths = [&]() {
        // Use explicit widths, or the median of input data for each process.
        if(override_homogeneous_kernel_widths) {
            if(override_homogeneous_kernel_widths.value.size() != data_points.nb_processes()) {
                throw std::runtime_error(fmt::format(
                    "Explicit kernel widths number does not match number of processes: expected {}, got {}",
                    data_points.nb_processes(),
                    override_homogeneous_kernel_widths.value.size()));
            }
            return std::move(override_homogeneous_kernel_widths.value);
        } else {
            auto widths = median_interval_widths(data_points);
            fmt::print(stderr, "Using deduced kernel widths: {}\n", fmt::join(widths, ", "));
            return widths;
        }
    };
    const auto width_to_interval_kernel = [](PointSpace w) { return IntervalKernel(w); };
    const auto width_to_interval_kernel_right_half = [](PointSpace w) { return IntervalKernel(w / 2., w / 2.); };

    if(config == KernelConfig::None) {
        return None{};
    } else if(config == KernelConfig::Homogeneous) {
        auto widths = get_homogeneous_kernel_widths();
        if(kernel_type == KernelType::Interval) {
            return map_to_vector(widths, width_to_interval_kernel);
        } else if(kernel_type == KernelType::IntervalRightHalf) {
            return map_to_vector(widths, width_to_interval_kernel_right_half);
        }
    } else if(config == KernelConfig::Heterogeneous) {
        if(kernel_type == KernelType::Interval) {
            return get_heterogeneous_kernels(data_points, width_to_interval_kernel);
        } else if(kernel_type == KernelType::IntervalRightHalf) {
            return get_heterogeneous_kernels(data_points, width_to_interval_kernel_right_half);
        }
    }
    // Catch unimplemented cases
    throw std::logic_error("Unknown kernel config/type combination");
}

int main(int argc, char * argv[]) {
    const Eigen::IOFormat eigen_format(Eigen::FullPrecision, 0, "\t"); // Matrix printing format, used later

    bool verbose = false;
    double gamma = 1.;
    double lambda = 1.;

    BaseOption base_option = None{}; // Base choice starts undefined and MUST be defined
    KernelConfig kernel_config = KernelConfig::None;
    KernelType kernel_type = KernelType::Interval; // Defaults to centered intervals
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
        [&base_option](string_view k_value, string_view delta_value) {
            const auto base_size = size_t(parse_strict_positive_int(k_value, "histogram K"));
            const auto delta = PointSpace(parse_strict_positive_double(delta_value, "histogram delta"));
            base_option = HistogramBase{base_size, delta};
        });
    parser.option2(
        {"haar"},
        "nb_scales",
        "delta",
        "Use Haar square wavelets (nb_scales > 0, delta > 0)",
        [&base_option](string_view k_value, string_view delta_value) {
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
            base_option = HaarBase{nb_scales, delta};
        });

    // Kernel setup
    parser.option(
        {"k", "kernel"},
        "none|homogeneous|heterogeneous",
        "Kernel configuration (default=none)",
        [&kernel_config](string_view value) {
            if(value == "none") {
                kernel_config = KernelConfig::None;
            } else if(value == "homogeneous") {
                kernel_config = KernelConfig::Homogeneous;
            } else if(value == "heterogeneous") {
                kernel_config = KernelConfig::Heterogeneous;
            } else {
                throw std::runtime_error(fmt::format("Unknown kernel configuration: '{}'", value));
            }
        });
    parser.option(
        {"t", "kernel-type"}, "interval|interval_right_half", "Kernel type", [&kernel_type](string_view value) {
            if(value == "interval") {
                kernel_type = KernelType::Interval;
            } else if(value == "interval_right_half") {
                kernel_type = KernelType::IntervalRightHalf;
            } else {
                throw std::runtime_error(fmt::format("Unknown kernel type: '{}'", value));
            }
        });
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

        // Check that base is set
        struct CheckBase {
            BaseType operator()(const None &) const { throw std::runtime_error("Function base is not defined"); }
            BaseType operator()(const HistogramBase & base) const { return base; }
            BaseType operator()(const HaarBase & base) const { return base; }
        };
        const BaseType base = visit(CheckBase{}, base_option);

        // Read input files
        const auto data_points = read_process_files(process_files);

        // Post processing: determine kernel setup, generate sorted points lists
        const auto post_processing_start = instant();
        const auto points = extract_point_lists(data_points);
        const auto kernels =
            determine_kernel_setup(data_points, kernel_config, kernel_type, override_homogeneous_kernel_widths);
        const auto post_processing_end = instant();
        fmt::print(
            stderr, "Post processing done: time = {}\n", duration_string(post_processing_end - post_processing_start));

        // Compute base/kernel specific values: B, G, B_hat
        // Ths visit() call generates an if/else if/.../else block for all combinations of base and kernel setup.
        // For each case, an overload of compute_intermediate_values indicated by its argument types does the
        // computation.
        const auto compute_b_g_start = instant();
        const CommonIntermediateValues intermediate_values = visit(
            [&points](const auto & base, const auto & kernels) {
                return compute_intermediate_values(points, base, kernels);
            },
            base,
            kernels);
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

            struct PrintBaseLine {
                void operator()(const HistogramBase & b) const {
                    fmt::print("# base = Histogram(K = {}, delta = {})\n", b.base_size, b.delta);
                }
                void operator()(const HaarBase & b) const {
                    fmt::print("# base = Haar(nb_scales = {}, delta = {})\n", b.nb_scales, b.delta);
                }
            };
            visit(PrintBaseLine{}, base);

            struct PrintKernelLine {
                string_view kernel_type_text; // Distinguish interval types which are merged in variant<...> type.

                void operator()(None) const { fmt::print("# kernels = None\n"); }
                void operator()(const std::vector<IntervalKernel> & kernels) const {
                    const auto widths = map_to_vector(kernels, [](IntervalKernel k) { return k.width; });
                    fmt::print("# kernels = homogeneous {} {{{}}}\n", kernel_type_text, fmt::join(widths, ", "));
                }
                void operator()(const HeterogeneousKernels<IntervalKernel> &) const {
                    // Do not print widths, too many to be useful
                    fmt::print("# kernels = heterogeneous {}\n", kernel_type_text);
                }

                PrintKernelLine(KernelType type) {
                    if(type == KernelType::Interval) {
                        kernel_type_text = "centered intervals";
                    } else if(type == KernelType::IntervalRightHalf) {
                        kernel_type_text = "intervals (right half)";
                    } else {
                        throw std::logic_error("unknown kernel type");
                    }
                }
            };
            visit(PrintKernelLine{kernel_type}, kernels);

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
