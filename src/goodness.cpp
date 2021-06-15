// Goodness computation tool, not needed for hawkes inferrence.
#include <cassert>
#include <memory>

#include "command_line.h"
#include "goodness.h"
#include "input.h"

Matrix_M_MK1 read_estimated_a_from(string_view filename, std::size_t nb_processes, std::size_t base_size) {
    try {
        auto estimated_a = Matrix_M_MK1(nb_processes, base_size);
        const auto start = instant();
        auto file = open_file(filename, "r");
        auto reader = LineByLineReader(file.get());
        auto get_next_non_comment_line = [&reader]() -> string_view {
            while(reader.read_next_line()) {
                string_view line = trim_ws(reader.current_line());
                if(!starts_with('#', line)) {
                    return line;
                }
            }
            throw std::runtime_error("expected line with values");
        };
        try {
            // Expected content is known, "demand driven" parser.
            {
                auto values_text = split('\t', get_next_non_comment_line());
                if(values_text.size() != nb_processes) {
                    throw std::runtime_error(
                        fmt::format("a_m,0: expected {} values, got {}", nb_processes, values_text.size()));
                }
                for(ProcessId m = 0; m < nb_processes; m += 1) {
                    estimated_a.set_0(m, parse_double(values_text[m], "a_{m,0}"));
                }
            }
            for(ProcessId l = 0; l < nb_processes; l += 1) {
                for(FunctionBaseId k = 0; k < base_size; k += 1) {
                    auto values_text = split('\t', get_next_non_comment_line());
                    if(values_text.size() != nb_processes) {
                        throw std::runtime_error(fmt::format(
                            "a_{{m,l={},k={}}}: expected {} values, got {}", l, k, nb_processes, values_text.size()));
                    }
                    for(ProcessId m = 0; m < nb_processes; m += 1) {
                        estimated_a.set_lk(m, l, k, parse_double(values_text[m], "a_{m,l,k}"));
                    }
                }
            }
        } catch(const std::runtime_error & e) {
            throw std::runtime_error(fmt::format("at line {}: {}", reader.current_line_number() + 1, e.what()));
        }
        const auto end = instant();
        fmt::print(stderr, "Estimated a loaded from '{}': time = {}\n", filename, duration_string(end - start));
        return estimated_a;
    } catch(const std::runtime_error & e) {
        throw std::runtime_error(fmt::format("Reading estimated a from {} ; {}", filename, e.what()));
    }
}

/******************************************************************************
 * Program entry point.
 */
int main(int argc, char * argv[]) {

    const auto command_line = CommandLineView(argc, argv);
    auto parser = CommandLineParser();

    std::unique_ptr<HistogramBase> base;
    std::vector<ProcessFile> process_files;
    string_view estimated_a_filename;
    string_view output_suffix = "lambda_hat";
    bool dump_region_info_option = false;

    constexpr auto undefined_override_tmax = -std::numeric_limits<Point>::infinity();
    Point override_tmax = undefined_override_tmax;

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

    parser.option(
        {"tmax"},
        "value",
        "Override tmax (default tmax[r] = max{union_m N_{m,r}})",
        [&override_tmax](string_view value) { override_tmax = Point(parse_double(value, "tmax override value")); });

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

    parser.positional(
        "estimated_a_filename", "Filename containing estimated a (inferrence output format)", [&](string_view value) {
            estimated_a_filename = value;
        });

    parser.option(
        {"s", "output-suffix"},
        "suffix",
        "lambda_hat values of file.bed stored in file.bed.<suffix>",
        [&](string_view value) { output_suffix = value; });

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
        const std::size_t base_size = base->base_size;

        // Read input files
        const ProcessFilesContent process_files_content = read_process_files(process_files);
        const DataByProcessRegion<SortedVec<Point>> points = extract_point_lists(process_files_content.points);
        const std::size_t nb_processes = points.nb_processes();
        const std::size_t nb_regions = points.nb_regions();
        const Matrix_M_MK1 estimated_a = read_estimated_a_from(estimated_a_filename, nb_processes, base_size);

        // Check tmax or determine default
        std::vector<Point> tmax;
        if(override_tmax != undefined_override_tmax) {
            // If overriden, use the same tmax for every region.
            // In practice it will only be used for one region, so this is ok.
            tmax = std::vector<Point>(nb_regions, override_tmax);
        } else {
            // Default tmax when not overriden by command line argument
            for(RegionId r = 0; r < nb_regions; r += 1) {
                Point tmax_r = -std::numeric_limits<Point>::infinity();
                for(ProcessId m = 0; m < nb_processes; m += 1) {
                    const SortedVec<Point> & sorted_points = points.data(m, r);
                    if(sorted_points.size() > 0) {
                        tmax_r = std::max(tmax_r, sorted_points[sorted_points.size() - 1]);
                    }
                }
                tmax.push_back(tmax_r);
            }
        }

        // Compute lambda hat values
        auto lambda_hat_values = DataByProcessRegion<std::vector<double>>(nb_processes, nb_regions);
        auto lambda_hat_tmax_values = DataByProcessRegion<double>(nb_processes, nb_regions);
        {
            const auto start = instant();
            for(RegionId r = 0; r < nb_regions; r += 1) {
                for(ProcessId m = 0; m < nb_processes; m += 1) {
                    lambda_hat_values.data(m, r) =
                        compute_lambda_hat_m_for_all_Nm(points.data_for_region(r), m, *base, estimated_a);
                    assert(lambda_hat_values.data(m, r).size() == points.data(m, r).size());

                    // There may be better options to compute lambda_hat_m(tmax[r]) if we had constraints on tmax.
                    // This recomputes the integrals only for tmax, but this is clean and has ok computation cost.
                    lambda_hat_tmax_values.data(m, r) =
                        compute_lambda_hat_m(points.data_for_region(r), m, *base, estimated_a, tmax[r]);
                }
            }
            const auto end = instant();
            fmt::print(stderr, "Computed lambda_hat values ; time = {}\n", duration_string(end - start));
        }

        // Output
        {
            const auto start = instant();
            // lambda_hat(points)
            for(ProcessId m = 0; m < nb_processes; m += 1) {
                std::string output_filename = fmt::format("{}.{}", process_files[m].filename, output_suffix);
                try {
                    auto file = open_file(output_filename, "w");
                    fmt::print(file.get(), "# region_name successive_lambda_hats_for_points\n");
                    for(RegionId r = 0; r < nb_regions; r += 1) {
                        const auto & region_name = process_files_content.region_names[r];
                        for(double v : lambda_hat_values.data(m, r)) {
                            fmt::print(file.get(), "{}\t{}\n", region_name, v);
                        }
                    }
                } catch(const std::runtime_error & e) {
                    throw std::runtime_error(fmt::format("Writing lambda_hats to {} ; {}", output_filename, e.what()));
                }
            }
            // lambda_hat(tmax)
            {
                std::string output_filename = fmt::format("tmax.{}", output_suffix);
                try {
                    auto file = open_file(output_filename, "w");
                    fmt::print(file.get(), "# region_name lambda_hat_{{m=0}}(tmax), ..., lambda_hat_{{m=M-1}}(tmax)\n");
                    for(RegionId r = 0; r < nb_regions; r += 1) {
                        fmt::print(
                            file.get(),
                            "{}\t{}\n",
                            process_files_content.region_names[r],
                            fmt::join(lambda_hat_tmax_values.data_for_region(r), "\t"));
                    }
                } catch(const std::runtime_error & e) {
                    throw std::runtime_error(
                        fmt::format("Writing lambda_hats(tmax) to {} ; {}", output_filename, e.what()));
                }
            }
            const auto end = instant();
            fmt::print(
                stderr,
                "Written output files (adding suffix '{}') ; time = {}\n",
                output_suffix,
                duration_string(end - start));
        }

    } catch(const CommandLineParser::Exception & exc) {
        fmt::print(stderr, "Error: {}. Use --help for a list of options.\n", exc.what());
        return EXIT_FAILURE;
    } catch(const std::exception & exc) {
        fmt::print(stderr, "Error: {}\n", exc.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}