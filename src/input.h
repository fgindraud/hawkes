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
struct BedFileRegions {
	struct BedFileRegion {
		std::vector<DataPoint> unsorted_points;

		// Lines where region was defined (starts at 0)
		std::size_t start_line;
		std::size_t end_line;
	};
	std::unordered_map<std::string, BedFileRegion> region_by_name; // region name -> region data

	std::size_t nb_regions () const { return region_by_name.size (); }

	static BedFileRegions read_from (std::FILE * file);
};

inline BedFileRegions BedFileRegions::read_from (std::FILE * file) {
	assert (file != nullptr);
	LineByLineReader reader (file);
	BedFileRegions regions;

	std::vector<DataPoint> current_region_points;
	std::string current_region_name;
	std::size_t current_region_start_line = 0;

	auto store_current_region = [&] () {
		if (!empty (current_region_name)) {
			// Store
			auto p = regions.region_by_name.emplace (std::move (current_region_name), BedFileRegion{
			                                                                              std::move (current_region_points),
			                                                                              current_region_start_line,
			                                                                              reader.current_line_number () - 1,
			                                                                          });
			assert (p.second);
			static_cast<void> (p); // Avoid unused var warning in non debug mode.
		}
	};

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
					throw std::runtime_error ("Interval bounds are invalid");
				}
				// Check if start of a new region
				if (region_name != current_region_name) {
					store_current_region ();
					// Setup new region
					current_region_points.clear ();
					current_region_name = to_string (region_name);
					current_region_start_line = reader.current_line_number ();
					if (empty (current_region_name)) {
						throw std::runtime_error ("Empty string as a region name");
					}
					auto it = regions.region_by_name.find (current_region_name);
					if (it != regions.region_by_name.end ()) {
						const auto & region = it->second;
						throw std::runtime_error (fmt::format ("Region '{}' already defined from line {} to {}."
						                                       " Regions must be defined in one contiguous block of lines.",
						                                       current_region_name, region.start_line + 1, region.end_line + 1));
					}
				}
				auto data_point = DataPoint{
				    (interval_start_position + interval_end_position) / 2.,
				    interval_end_position - interval_start_position,
				};
				current_region_points.emplace_back (data_point);
			}
		}
		store_current_region (); // Add last region
		return regions;
	} catch (const std::runtime_error & e) {
		// Add some context to an error.
		throw std::runtime_error (
		    fmt::format ("Parsing BED file at line {}: {}", reader.current_line_number () + 1, e.what ()));
	}
}
