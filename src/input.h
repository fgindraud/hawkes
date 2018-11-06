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
 * BED format parsing.
 */
