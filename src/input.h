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

#include "types.h"
#include "utils.h"

/******************************************************************************
 * File parsing utils.
 */

// Custom deleters for unique_ptr
struct FreeDeleter {
	void operator() (void * ptr) noexcept { std::free (ptr); }
};
struct FcloseDeleter {
	void operator() (std::FILE * f) noexcept {
		std::fclose (f); // Ignore fclose errors
	}
};

/* Open file with error checking.
 * On error, an exception is thrown.
 * Same arguments as fopen.
 */
inline std::unique_ptr<std::FILE, FcloseDeleter> open_file (string_view pathname, string_view mode) {
	// fopen requires null terminated strings.
	errno = 0;
	std::unique_ptr<std::FILE, FcloseDeleter> file (
	    std::fopen (to_string (pathname).c_str (), to_string (mode).c_str ()));
	if (!file) {
		throw std::system_error (errno, std::system_category ());
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
	not_null<std::FILE *> input_;

	std::unique_ptr<char, FreeDeleter> current_line_data_{nullptr};
	std::size_t current_line_data_buffer_size_{0};
	std::size_t current_line_size_{0};

	std::size_t lines_read_{0};

public:
	// State: no stored data
	explicit LineByLineReader (not_null<std::FILE *> input) : input_ (input) {}

	/* Read a line from the file.
	 * Returns true if a line was read, false on eof.
	 * Throws an exception on error.
	 */
	bool read_next_line ();

	// Access current line as const
	string_view current_line () const {
		if (current_line_data_) {
			return string_view (current_line_data_.get (), current_line_size_);
		} else {
			throw std::runtime_error ("LineByLineReader: no line data available");
		}
	}

	// Counts from 0
	std::size_t current_line_number () const { return lines_read_ - 1; }

	bool eof () const { return std::feof (input_); }
};

inline bool LineByLineReader::read_next_line () {
	char * buf_ptr = current_line_data_.release ();
	errno = 0;
	auto r = ::getline (&buf_ptr, &current_line_data_buffer_size_, input_);
	auto error_code = errno;
	current_line_data_.reset (buf_ptr);

	if (r >= 0) {
		current_line_size_ = static_cast<std::size_t> (r);
		++lines_read_;
		return true;
	} else {
		if (std::feof (input_)) {
			current_line_data_.reset (); // Clear buffer so that current_line will fail.
			return false;
		} else {
			throw std::system_error (error_code, std::system_category ());
		}
	}
}

/******************************************************************************
 * Parsing utils.
 */

// Add process to table with checking of region number TODO rm
template <typename DataType>
inline ProcessId ProcessesData<DataType>::add_process (string_view name,
                                                       std::vector<ProcessRegionData<DataType>> && regions) {
	if (nb_processes () == 0) {
		// First process defines the number of regions
		process_regions_ = Vector2d<ProcessRegionData<DataType>> (0, regions.size ());
	}
	if (int(regions.size ()) != nb_regions ()) {
		throw std::runtime_error (
		    fmt::format ("Adding process data: expected {} regions, got {}", nb_regions (), regions.size ()));
	}
	const auto new_process_id = ProcessId{nb_processes ()};
	process_regions_.append_row (std::move (regions));
	process_names_.emplace_back (to_string (name));
	return new_process_id;
}

/******************************************************************************
 * BED format parsing.
 */
inline std::vector<RawRegionData> read_all_from_bed_file (not_null<FILE *> file) {
	std::vector<RawRegionData> regions;
	LineByLineReader reader (file);

	std::vector<PointInterval> current_region_intervals;
	std::string current_region_name;

	try {
		while (reader.read_next_line ()) {
			const string_view line = trim_ws (reader.current_line ());
			if (starts_with ('#', line)) {
				// Comment or header, ignore
			} else {
				const auto fields = split ('\t', line);
				if (!(fields.size () >= 3)) {
					throw std::runtime_error ("Line must contain at least 3 fields: (region, start, end)");
				}
				const string_view region_name = fields[0];
				const Point interval_start_position = parse_int (fields[1], "interval_position_start");
				const Point interval_end_position = parse_int (fields[2], "interval_position_end");
				if (!(interval_start_position <= interval_end_position)) {
					throw std::runtime_error ("interval bounds are invalid");
				}
				// Check is start of a new region
				if (region_name != current_region_name) {
					if (empty (region_name)) {
						throw std::runtime_error ("empty string as a region name");
					}
					if (!empty (current_region_name)) {
						// End current region and store its data
						regions.emplace_back (RawRegionData{std::move (current_region_name), std::move (current_region_intervals)});
					}
					current_region_intervals.clear ();
					current_region_name = to_string (region_name);
				}
				current_region_intervals.emplace_back (PointInterval{interval_start_position, interval_end_position});
			}
		}
		return regions;
	} catch (const std::runtime_error & e) {
		// Add some context to an error.
		throw std::runtime_error (
		    fmt::format ("Parsing BED file at line {}: {}", reader.current_line_number () + 1, e.what ()));
	}
}

inline std::vector<RawRegionData> read_selected_from_bed_file (not_null<FILE *> file,
                                                               span<const string_view> region_names) {
	// Read all regions data, then copy them in the right order
	// We must copy in the case of duplicated region names in the list.
	std::vector<RawRegionData> all_regions = read_all_from_bed_file (file);
	std::vector<RawRegionData> selected_regions;
	selected_regions.reserve (region_names.size ());
	for (const string_view name : region_names) {
		auto it = std::find_if (all_regions.begin (), all_regions.end (),
		                        [name](const auto & element) { return element.name == name; });
		if (it != all_regions.end ()) {
			selected_regions.emplace_back (*it);
		} else {
			throw std::runtime_error (fmt::format ("Selected region was not found: {}", name));
		}
	}
	return selected_regions;
}
