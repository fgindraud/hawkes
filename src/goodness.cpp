#include <cassert>
#include <memory>

#include "command_line.h"
#include "goodness.h"
#include "input.h"

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