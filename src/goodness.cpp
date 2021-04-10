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
            throw std::runtime_error("Expected line with values");
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
            throw std::runtime_error(
                fmt::format("Parsing estimated a at line {}: {}", reader.current_line_number() + 1, e.what()));
        }
        const auto end = instant();
        fmt::print(stderr, "Estimated a loaded from '{}': time = {}\n", filename, duration_string(end - start));
        return estimated_a;
    } catch(const std::runtime_error & e) {
        throw std::runtime_error(fmt::format("Reading estimated a from {}: {}", filename, e.what()));
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

    parser.positional(
        "estimated_a_filename", "Filename containing estimated a (inferrence output format)", [&](string_view value) {
            estimated_a_filename = value;
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
        auto estimated_a = read_estimated_a_from(estimated_a_filename, points.nb_processes(), base->base_size);

        fmt::print("{}\n", estimated_a.inner);

    } catch(const CommandLineParser::Exception & exc) {
        fmt::print(stderr, "Error: {}. Use --help for a list of options.\n", exc.what());
        return EXIT_FAILURE;
    } catch(const std::exception & exc) {
        fmt::print(stderr, "Error: {}\n", exc.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}