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
		fmt::print (stderr, "Process {} loaded: regions = {} ; time = {}\n", filename, regions.size (),
		            duration_string (end - start));
		// TODO information on points ?
		return regions;
	} catch (const std::runtime_error & e) {
		throw std::runtime_error (fmt::format ("Reading process data from {}: {}", filename, e.what ()));
	}
}

/******************************************************************************
 * Compute kernel widths as average of interval sizes.
 */
static PointSpace average_interval_width (const RawProcessData & raw_process) {
	int64_t sum_of_interval_widths = 0;
	int64_t nb_elements = 0;
	for (const auto & region : raw_process.regions) {
		nb_elements += int64_t (region.unsorted_intervals.size ());
		for (const auto & interval : region.unsorted_intervals) {
			assert (interval.right >= interval.left);
			sum_of_interval_widths += interval.right - interval.left;
		}
	}
	if (nb_elements == 0) {
		throw std::runtime_error ("Kernel width deduction: process contains no points");
	}
	return PointSpace (sum_of_interval_widths / nb_elements);
}
static std::vector<PointSpace> average_interval_widths (const std::vector<RawProcessData> & raw_processes) {
	std::vector<PointSpace> widths;
	widths.reserve (raw_processes.size ());
	for (const auto & raw_process : raw_processes) {
		widths.emplace_back (average_interval_width (raw_process));
	}
	return widths;
}

/******************************************************************************
 * Program entry point.
 */
int main (int argc, char * argv[]) {
	bool verbose = false;
	double gamma = 3.;

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

	parser.flag ({"v", "verbose"}, "Enable verbose output", [&]() { verbose = true; });

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

		const auto nb_processes = raw_processes.size ();

		const auto kernels = [&]() -> variant<None, std::vector<IntervalKernel>> {
			// Helper: choose the source of kernel widths (explicit or deduced).
			auto get_kernel_widths = [&]() -> std::vector<PointSpace> {
				if (explicit_kernel_widths) {
					if (explicit_kernel_widths.value.size () != nb_processes) {
						throw std::runtime_error (
						    fmt::format ("Explicit kernel widths number does not match number of processes: expected {}, got {}",
						                 nb_processes, explicit_kernel_widths.value.size ()));
					}
					return std::move (explicit_kernel_widths.value);
				} else {
					return average_interval_widths (raw_processes);
				}
			};

			if (use_kernel == Kernel::Interval) {
				std::vector<IntervalKernel> kernels;
				kernels.reserve (nb_processes);
				for (PointSpace width : get_kernel_widths ()) {
					kernels.emplace_back (IntervalKernel{width});
				}
				return kernels;
			} else {
				return None{};
			}
		}();

		const auto point_processes = ProcessesRegionData::from_raw (raw_processes);

		const auto intermediate_values = visit (
		    [&](const auto & base, const auto & kernels) {
			    // Code to compute intermediate values for each combination of base and kernels.
			    return compute_intermediate_values (point_processes, base, kernels);
		    },
		    base, kernels);

		const auto lasso_parameters = compute_lasso_parameters (intermediate_values, gamma);
		const auto estimated_a = compute_estimated_a_with_lasso (lasso_parameters);

		// Print results
		if (verbose) {
			// Header
			fmt::print ("# Processes = {{\n");
			for (size_t i = 0; i < nb_processes; ++i) {
				const auto & p = raw_processes[i];
				const string_view suffix = p.direction == RawProcessData::Direction::Backward ? " (backward)" : "";
				fmt::print ("#  [{}] {}{}\n", i, p.name, suffix);
			}
			fmt::print ("# }}\n");

			struct PrintBaseLine {
				void operator() (None) const {}
				void operator() (HistogramBase b) const {
					fmt::print ("# base = Histogram(K = {}, delta = {})\n", b.base_size, b.delta);
				}
			};
			visit (PrintBaseLine{}, base);

			struct PrintKernelLine {
				void operator() (None) const { fmt::print ("# kernels = None\n"); }
				void operator() (const std::vector<IntervalKernel> & kernels) const {
					fmt::print ("# kernels = Intervals{{}}\n"); // FIXME
				}
			};
			visit (PrintKernelLine{}, kernels);

			fmt::print ("# gamma = {}\n", gamma);

			fmt::print ("# Rows = {{0}} U {{(l,k)}} (order = 0,(0,0),..,(0,K-1),(1,0),..,(1,K-1),...,(M-1,K-1))\n");
			fmt::print ("# Columns = {{m}}\n");
		}
		const Eigen::IOFormat eigen_format (Eigen::FullPrecision, 0, "\t");
		fmt::print ("{}\n", estimated_a.inner.format (eigen_format));

	} catch (const CommandLineParser::Exception & exc) {
		fmt::print (stderr, "Error: {}. Use --help for a list of options.\n", exc.what ());
		return EXIT_FAILURE;
	} catch (const std::exception & exc) {
		fmt::print (stderr, "Error: {}\n", exc.what ());
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
