#pragma once
// File parsing

#include <algorithm>
#include <cassert>
#include <cerrno>  // LineByLineReader
#include <cstdio>  // LineByLineReader io
#include <cstdlib> // free
#include <memory>  // open_file
#include <stdexcept>
#include <string>
#include <system_error> // io errors
#include <unordered_map>
#include <unordered_set>

#include "types.h"
#include "utils.h"

/******************************************************************************
 * File parsing utils.
 */

// Custom deleters for unique_ptr
struct FreeDeleter {
    void operator()(void * ptr) noexcept { std::free(ptr); }
};
struct FcloseDeleter {
    void operator()(std::FILE * f) noexcept {
        std::fclose(f); // Ignore fclose errors
    }
};

/* Open file with error checking.
 * On error, an exception is thrown.
 * Same arguments as fopen.
 */
inline std::unique_ptr<std::FILE, FcloseDeleter> open_file(string_view pathname, string_view mode) {
    // fopen requires null terminated strings.
    errno = 0;
    std::unique_ptr<std::FILE, FcloseDeleter> file(std::fopen(to_string(pathname).c_str(), to_string(mode).c_str()));
    if(!file) {
        throw std::system_error(errno, std::system_category());
    }
    return file;
}

/* LineByLineReader allows to read a file line by line.
 * After construction, the reader has no stored data.
 * The user should call read_next_line to extract a line from the file.
 * The line is stored in an internal buffer, and is valid until the next call to read_next_line.
 */
class LineByLineReader {
  private:
    std::FILE * input_;

    std::unique_ptr<char, FreeDeleter> current_line_data_{nullptr};
    std::size_t current_line_data_buffer_size_{0};
    std::size_t current_line_size_{0};

    std::size_t lines_read_{0};

  public:
    // State: no stored data
    explicit LineByLineReader(std::FILE * input) : input_(input) { assert(input != nullptr); }

    /* Read a line from the file.
     * Returns true if a line was read, false on eof.
     * Throws an exception on error.
     */
    bool read_next_line();

    // Access current line as const
    string_view current_line() const {
        if(current_line_data_) {
            return string_view(current_line_data_.get(), current_line_size_);
        } else {
            throw std::runtime_error("LineByLineReader: no line data available");
        }
    }

    // Counts from 0
    std::size_t current_line_number() const { return lines_read_ - 1; }

    bool eof() const { return std::feof(input_); }
};

inline bool LineByLineReader::read_next_line() {
    char * buf_ptr = current_line_data_.release();
    errno = 0;
    auto r = ::getline(&buf_ptr, &current_line_data_buffer_size_, input_);
    auto error_code = errno;
    current_line_data_.reset(buf_ptr);

    if(r >= 0) {
        current_line_size_ = static_cast<std::size_t>(r);
        ++lines_read_;
        return true;
    } else {
        if(std::feof(input_)) {
            current_line_data_.reset(); // Clear buffer so that current_line will fail.
            return false;
        } else {
            throw std::system_error(error_code, std::system_category());
        }
    }
}

/******************************************************************************
 * BED format parsing.
 */
struct BedFileRegions {
    struct BedFileRegion {
        std::vector<DataPoint> unsorted_points;

        // Lines where region was defined (starts at 0)
        std::size_t start_line;
        std::size_t end_line;
    };
    std::unordered_map<std::string, BedFileRegion> region_by_name; // region name -> region data

    std::size_t nb_regions() const { return region_by_name.size(); }

    static BedFileRegions read_from(std::FILE * file);
};

inline BedFileRegions BedFileRegions::read_from(std::FILE * file) {
    assert(file != nullptr);
    LineByLineReader reader(file);
    BedFileRegions regions;

    std::vector<DataPoint> current_region_points;
    std::string current_region_name;
    std::size_t current_region_start_line = 0;

    auto store_current_region = [&]() {
        if(!empty(current_region_name)) {
            // Store
            auto p = regions.region_by_name.emplace(
                std::move(current_region_name),
                BedFileRegion{
                    std::move(current_region_points),
                    current_region_start_line,
                    reader.current_line_number() - 1,
                });
            assert(p.second);
            static_cast<void>(p); // Avoid unused var warning in non debug mode.
        }
    };

    try {
        while(reader.read_next_line()) {
            const string_view line = trim_ws(reader.current_line());
            if(starts_with('#', line)) {
                // Comment or header, ignore
            } else {
                const auto fields = split_first_n<3>('\t', line);
                if(!fields) {
                    throw std::runtime_error("Line must contain at least 3 fields: (region, start, end)");
                }
                const string_view region_name = fields.value[0];
                const Point interval_start_position = parse_double(fields.value[1], "interval_position_start");
                const Point interval_end_position = parse_double(fields.value[2], "interval_position_end");
                if(!(interval_start_position <= interval_end_position)) {
                    throw std::runtime_error("Interval bounds are invalid");
                }
                // Check if start of a new region
                if(region_name != current_region_name) {
                    store_current_region();
                    // Setup new region
                    current_region_points.clear();
                    current_region_name = to_string(region_name);
                    current_region_start_line = reader.current_line_number();
                    if(empty(current_region_name)) {
                        throw std::runtime_error("Empty string as a region name");
                    }
                    auto it = regions.region_by_name.find(current_region_name);
                    if(it != regions.region_by_name.end()) {
                        const auto & region = it->second;
                        throw std::runtime_error(fmt::format(
                            "Region '{}' already defined from line {} to {}."
                            " Regions must be defined in one contiguous block of lines.",
                            current_region_name,
                            region.start_line + 1,
                            region.end_line + 1));
                    }
                }
                auto data_point = DataPoint{
                    (interval_start_position + interval_end_position) / 2.,
                    interval_end_position - interval_start_position,
                };
                current_region_points.emplace_back(data_point);
            }
        }
        store_current_region(); // Add last region
        return regions;
    } catch(const std::runtime_error & e) {
        // Add some context to an error.
        throw std::runtime_error(
            fmt::format("Parsing BED file at line {}: {}", reader.current_line_number() + 1, e.what()));
    }
}

/******************************************************************************
 * File parsing, high level, with info messages.
 */
enum class ProcessDirection {
    Forward,  // Keep points coordinates
    Backward, // x -> -x for all points coordinates
};

struct ProcessFile {
    string_view filename;
    ProcessDirection direction;
};

inline BedFileRegions read_regions_from(string_view filename) {
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

inline std::vector<std::string> union_of_all_region_names(const std::vector<BedFileRegions> & bed_files_content) {
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

struct ProcessFilesContent {
    std::vector<std::string> region_names;
    DataByProcessRegion<SortedVec<DataPoint>> points;
};

inline ProcessFilesContent read_process_files(const std::vector<ProcessFile> & files) {
    if(files.empty()) {
        throw std::runtime_error("read_process_files: process list is empty");
    }
    auto bed_files_content = map_to_vector(files, [](const ProcessFile & f) { return read_regions_from(f.filename); });
    std::vector<std::string> all_region_names = union_of_all_region_names(bed_files_content);

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
    return {std::move(all_region_names), std::move(points)};
}

// Print a table with region statistics to stdout. Used for checking data parsing.
inline void print_region_info(const std::vector<ProcessFile> & files) {
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

/******************************************************************************
 * Post processing.
 */
inline DataByProcessRegion<SortedVec<Point>> extract_point_lists(
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

inline std::vector<PointSpace> median_interval_widths(const DataByProcessRegion<SortedVec<DataPoint>> & data_points) {
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

template <typename KT> static HomogeneousKernels<KT> determine_homogeneous_kernels(
    const DataByProcessRegion<SortedVec<DataPoint>> & data_points,
    const Optional<std::vector<PointSpace>> & override_homogeneous_kernel_widths) {

    auto kernels_from_widths = [](const std::vector<PointSpace> & widths) {
        return HomogeneousKernels<KT>{map_to_vector(widths, [](PointSpace width) { return KT(width); })};
    };

    if(override_homogeneous_kernel_widths) {
        // Use explicit widths if provided
        if(override_homogeneous_kernel_widths.value.size() != data_points.nb_processes()) {
            throw std::runtime_error(fmt::format(
                "Explicit kernel widths number does not match number of processes: expected {}, got {}",
                data_points.nb_processes(),
                override_homogeneous_kernel_widths.value.size()));
        }
        return kernels_from_widths(override_homogeneous_kernel_widths.value);
    } else {
        // Use median widths by default
        auto widths = median_interval_widths(data_points);
        fmt::print(stderr, "Using deduced kernel widths: {}\n", fmt::join(widths, ", "));
        return kernels_from_widths(widths);
    }
}

template <typename KT> inline HeterogeneousKernels<KT> extract_heterogeneous_kernels(
    const DataByProcessRegion<SortedVec<DataPoint>> & data_points) {
    const auto nb_processes = data_points.nb_processes();
    const auto nb_regions = data_points.nb_regions();

    DataByProcessRegion<std::vector<KT>> kernels(nb_processes, nb_regions);
    std::vector<KT> maximum_width_kernels;

    for(ProcessId m = 0; m < nb_processes; ++m) {
        PointSpace max_width = 0.;
        for(RegionId r = 0; r < nb_regions; ++r) {
            const auto & region_data_points = data_points.data(m, r);
            std::vector<KT> region_kernels;
            region_kernels.reserve(region_data_points.size());
            for(const auto & data_point : region_data_points) {
                const auto width = fix_zero_width(data_point.width);
                region_kernels.emplace_back(width);
                max_width = std::max(max_width, width);
            }
            kernels.data(m, r) = std::move(region_kernels);
        }
        maximum_width_kernels.emplace_back(max_width);
    }
    return {std::move(kernels), std::move(maximum_width_kernels)};
}