#include <algorithm>
#include <chrono>
#include <string>

#include <cassert>
#include <random>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "command_line.h"
#include "computations.h"
#include "input.h"
#include "lassoshooting.h"
#include "utils.h"

/******************************************************************************
 * Time measurement primitives.
 */
template <typename Rep, typename Period>
static std::string duration_string (std::chrono::duration<Rep, Period> duration) {
	namespace chrono = std::chrono;
	using chrono::duration_cast;
	const auto hours = duration_cast<chrono::hours> (duration).count ();
	if (hours > 10) {
		return fmt::format ("{}h", hours);
	}
	const auto minutes = duration_cast<chrono::minutes> (duration).count ();
	if (minutes > 10) {
		return fmt::format ("{}m", minutes);
	}
	const auto seconds = duration_cast<chrono::seconds> (duration).count ();
	if (seconds > 10) {
		return fmt::format ("{}s", seconds);
	}
	const auto milliseconds = duration_cast<chrono::milliseconds> (duration).count ();
	if (milliseconds > 10) {
		return fmt::format ("{}ms", milliseconds);
	}
	const auto microseconds = duration_cast<chrono::microseconds> (duration).count ();
	if (microseconds > 10) {
		return fmt::format ("{}us", microseconds);
	}
	const auto nanoseconds = duration_cast<chrono::nanoseconds> (duration).count ();
	return fmt::format ("{}ns", nanoseconds);
}

static std::chrono::high_resolution_clock::time_point instant () {
	return std::chrono::high_resolution_clock::now ();
}

/******************************************************************************
 * Reading process data.
 */
static std::vector<RawRegionData> read_regions_from (string_view filename, span<const string_view> region_names) {
	try {
		const auto start = instant ();
		auto file = open_file (filename, "r");
		auto regions = read_selected_from_bed_file (file.get (), region_names);
		const auto end = instant ();
		fmt::print (stderr, "Process loaded from {}: regions = {} ; time = {}\n", filename, regions.size (),
		            duration_string (end - start));
		return regions;
	} catch (const std::runtime_error & e) {
		throw std::runtime_error (fmt::format ("Reading process data from {}: {}", filename, e.what ()));
	}
}

/******************************************************************************
 * Program entry point.
 */
#include <iostream> //FIXME debug output

int main (int argc, char * argv[]) {
	double gamma = 3.;

	struct None {};
	variant<None, HistogramBase> base = None{};

	enum class Kernel { None, Interval };
	Kernel use_kernel = Kernel::None;
	Optional<std::vector<PointSpace>> explicit_kernel_widths;

	std::vector<string_view> current_region_names;
	std::vector<RawProcessData> raw_processes;

	// Command line parsing setup
	const auto command_line = CommandLineView (argc, argv);
	auto parser = CommandLineParser ();

	parser.flag ({"h", "help"}, "Display this help", [&]() { //
		parser.usage (stderr, command_line.program_name ());
		std::exit (EXIT_SUCCESS);
	});

	parser.option ({"g", "gamma"}, "value", "Set gamma value (double, positive)", [&gamma](string_view value) { //
		gamma = parse_strict_positive_double (value, "gamma");
	});

	parser.option2 ({"histogram"}, "K", "delta", "Use an histogram base (k > 0, delta > 0)",
	                [&base](string_view k_value, string_view delta_value) {
		                const auto base_size = size_t (parse_strict_positive_int (k_value, "histogram K"));
		                const auto delta = PointSpace (parse_strict_positive_int (delta_value, "histogram delta"));
		                base = HistogramBase{base_size, delta};
	                });

	parser.option ({"kernel"}, "none|interval", "Use a kernel type (default=none)", [&use_kernel](string_view value) {
		if (value == "none") {
			use_kernel = Kernel::None;
		} else if (value == "interval") {
			use_kernel = Kernel::Interval;
		} else {
			throw std::runtime_error (fmt::format ("Unknown kernel type option: '{}'", value));
		}
	});
	parser.option ({"kernel-widths"}, "w0[:w1:w2:...]", "Use explicit kernel widths (default=deduced)",
	               [&explicit_kernel_widths](string_view values) {
		               std::vector<PointSpace> widths;
		               for (string_view value : split (':', values)) {
			               widths.emplace_back (PointSpace (parse_strict_positive_int (value, "kernel width")));
		               }
		               explicit_kernel_widths = std::move (widths);
	               });

	parser.option ({"r", "regions"}, "r0[,r1,r2,...]", "Set region names extracted from next files",
	               [&current_region_names](string_view regions) {
		               auto region_names = split (',', regions);
		               if (region_names.empty ()) {
			               throw std::runtime_error ("List of region names is empty");
		               }
		               if (!current_region_names.empty () && current_region_names.size () != region_names.size ()) {
			               throw std::runtime_error ("New region name set must have the same length as all previous ones");
		               }
		               current_region_names = std::move (region_names);
	               });
	auto add_process_from_file = [&current_region_names, &raw_processes](string_view filename,
	                                                                     RawProcessData::Direction direction) {
		if (current_region_names.empty ()) {
			throw std::runtime_error ("List of region names is empty: set region names before reading a file");
		}
		auto regions = read_regions_from (filename, make_span (current_region_names));
		assert (regions.size () == current_region_names.size ());
		raw_processes.emplace_back (RawProcessData{to_string (filename), std::move (regions), direction});
	};
	parser.option ({"f", "file-forward"}, "filename", "Add process regions from file",
	               [add_process_from_file](string_view filename) {
		               add_process_from_file (filename, RawProcessData::Direction::Forward);
	               });
	parser.option ({"b", "file-backward"}, "filename", "Add process regions (reversed) from file",
	               [add_process_from_file](string_view filename) {
		               add_process_from_file (filename, RawProcessData::Direction::Backward);
	               });

	try {
		// Parse command line arguments. All actions declared to the parser will be called here.
		parser.parse (command_line);

		// TODO Check that base is correctly defined.

		const auto point_processes = ProcessesRegionData::from_raw (raw_processes);

		// TODO Deduce kernel widths if not provided
		// Check kernel numbers if provided.

		// TEST
		auto histogram = get<HistogramBase> (base);

		auto values = compute_intermediate_values (point_processes, histogram);
		auto parameters = compute_lasso_parameters (values, gamma);

		assert (parameters.sum_of_b.inner.allFinite ());
		assert (parameters.sum_of_g.inner.allFinite ());
		assert (parameters.d.inner.allFinite ());

		Matrix_M_MK1 estimated_a (point_processes.nb_processes (), histogram.base_size);
		for (ProcessId m = 0; m < point_processes.nb_processes (); ++m) {
			estimated_a.values_for_m (m) = lassoshooting (parameters.sum_of_g.inner, parameters.sum_of_b.values_for_m (m),
			                                              parameters.d.values_for_m (m), 1.);
		}

		std::cout << estimated_a.inner << "\n";

	} catch (const CommandLineParser::Exception & exc) {
		fmt::print (stderr, "Error: {}. Use --help for a list of options.\n", exc.what ());
		return EXIT_FAILURE;
	} catch (const std::exception & exc) {
		fmt::print (stderr, "Error: {}\n", exc.what ());
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
