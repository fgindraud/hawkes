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
	std::FILE * input_;

	std::unique_ptr<char, FreeDeleter> current_line_data_{nullptr};
	std::size_t current_line_data_buffer_size_{0};
	std::size_t current_line_size_{0};

	std::size_t lines_read_{0};

public:
	// State: no stored data
	explicit LineByLineReader (std::FILE * input) : input_ (input) { assert (input != nullptr); }

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
 * BED format parsing.
 */
struct BedRegion {
	std::string name;
	std::vector<PointInterval> unsorted_intervals;
};

inline std::vector<BedRegion> read_all_from_bed_file (std::FILE * file) {
	assert (file != nullptr);
	std::vector<BedRegion> regions;
	LineByLineReader reader (file);

	std::vector<PointInterval> current_region_intervals;
	std::string current_region_name;

	try {
		while (reader.read_next_line ()) {
			const string_view line = trim_ws (reader.current_line ());
			if (starts_with ('#', line)) {
				// Comment or header, ignore
			} else {
				const auto fields = split_first_n<3> ('\t', line);
				if (!fields) {
					throw std::runtime_error ("Line must contain at least 3 fields: (region, start, end)");
				}
				const string_view region_name = fields.value[0];
				const Point interval_start_position = parse_double (fields.value[1], "interval_position_start");
				const Point interval_end_position = parse_double (fields.value[2], "interval_position_end");
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
						regions.emplace_back (BedRegion{std::move (current_region_name), std::move (current_region_intervals)});
					}
					current_region_intervals.clear ();
					current_region_name = to_string (region_name);
				}
				auto point_interval = PointInterval{
				    (interval_start_position + interval_end_position) / 2.,
				    interval_end_position - interval_start_position,
				};
				current_region_intervals.emplace_back (point_interval);
			}
		}
		return regions;
	} catch (const std::runtime_error & e) {
		// Add some context to an error.
		throw std::runtime_error (
		    fmt::format ("Parsing BED file at line {}: {}", reader.current_line_number () + 1, e.what ()));
	}
}

inline std::vector<BedRegion> read_selected_from_bed_file (std::FILE * file,
                                                           const std::vector<string_view> & region_names) {
	assert (file != nullptr);
	// Read all regions data, then copy them in the right order to the final vector of regions
	std::vector<BedRegion> all_regions = read_all_from_bed_file (file);
	std::vector<BedRegion> selected_regions;
	selected_regions.reserve (region_names.size ());
	for (auto name_it = region_names.begin (); name_it != region_names.end (); ++name_it) {
		const auto wanted_name = *name_it;
		auto it = std::find_if (all_regions.begin (), all_regions.end (),
		                        [wanted_name](const auto & region) { return region.name == wanted_name; });
		if (it != all_regions.end ()) {
			const bool is_wanted_name_required_afterward =
			    std::any_of (name_it + 1, region_names.end (),
			                 [wanted_name](const string_view other_region_name) { return wanted_name == other_region_name; });
			if (!is_wanted_name_required_afterward) {
				selected_regions.emplace_back (std::move (*it)); // We can move data (avoid a copy).
			} else {
				selected_regions.emplace_back (*it);
			}
		} else {
			throw std::runtime_error (fmt::format ("Required region was not found: {}", wanted_name));
		}
	}
	return selected_regions;
}
