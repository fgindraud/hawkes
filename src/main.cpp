#include <algorithm>
#include <chrono>
#include <string>

#include <cassert>
#include <random>

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
 * Program entry point.
 */

enum class KernelConfig {
	None,          // No Kernel, "ponctual" mode
	Homogeneous,   // One kernel per process, chosen as median of BED interval widths by default
	Heterogeneous, // One kernel per point, chosen as width of the BED interval from which the point was generated
};
enum class KernelType {
	Interval,          // L2-normalized indicator function centered on 0
	IntervalRightHalf, // L2-normalized indicator function on [0, width/2]
};

using BaseOption = variant<None, HistogramBase>;
using BaseType = variant<HistogramBase>;

enum class ProcessDirection {
	Forward,  // Keep points coordinates
	Backward, // x -> -x for all points coordinates
};

struct ProcessFile {
	string_view filename;
	std::vector<string_view> regions_to_extract;
	ProcessDirection direction;
};

static std::vector<BedRegion> read_regions_from (string_view filename, const std::vector<string_view> & region_names) {
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

static DataByProcessRegion<SortedVec<PointInterval>> read_process_files (const std::vector<ProcessFile> & files) {
	if (files.empty ()) {
		throw std::runtime_error ("read_process_files: process list is empty");
	}
	const auto nb_processes = files.size ();
	const auto nb_regions = files[0].regions_to_extract.size ();
	DataByProcessRegion<SortedVec<PointInterval>> intervals (nb_processes, nb_regions);

	for (ProcessId m = 0; m < nb_processes; m++) {
		const auto & file = files[m];
		auto points_by_region = read_regions_from (file.filename, file.regions_to_extract);
		if (points_by_region.size () != nb_regions) {
			throw std::runtime_error (
			    fmt::format ("read_process_files: process {} has wrong region number: got {}, expected {}", m,
			                 points_by_region.size (), nb_regions));
		}
		for (RegionId r = 0; r < nb_regions; ++r) {
			// Apply reversing if requested before sorting them in increasing order
			if (file.direction == ProcessDirection::Backward) {
				for (auto & interval : points_by_region[r].unsorted_intervals) {
					interval.center = -interval.center;
				}
			}
			intervals.data (m, r) =
			    SortedVec<PointInterval>::from_unsorted (std::move (points_by_region[r].unsorted_intervals));
		}
	}
	return intervals;
}

static DataByProcessRegion<SortedVec<Point>>
extract_point_lists (const DataByProcessRegion<SortedVec<PointInterval>> & intervals) {
	const auto nb_processes = intervals.nb_processes ();
	const auto nb_regions = intervals.nb_regions ();
	DataByProcessRegion<SortedVec<Point>> points (nb_processes, nb_regions);
	for (ProcessId m = 0; m < nb_processes; ++m) {
		for (RegionId r = 0; r < nb_regions; ++r) {
			const auto & region_intervals = intervals.data (m, r);
			const auto region_size = region_intervals.size ();
			std::vector<Point> region_points (region_size);
			for (size_t i = 0; i < region_size; ++i) {
				region_points[i] = region_intervals[i].center;
			}
			points.data (m, r) = SortedVec<Point>::from_sorted (std::move (region_points));
		}
	}
	return points;
}

static std::vector<PointSpace>
median_interval_widths (const DataByProcessRegion<SortedVec<PointInterval>> & intervals) {
	const auto compute_median_width = [&intervals](ProcessId m) {
		const auto sum_of_region_sizes = [&intervals, m]() {
			size_t sum = 0;
			for (RegionId r = 0; r < intervals.nb_regions (); ++r) {
				sum += intervals.data (m, r).size ();
			}
			return sum;
		}();
		// Degenerate case
		if (sum_of_region_sizes == 0) {
			return PointSpace (0);
		}
		// Build a vector containing the unsorted widths of all intervals from all regions
		std::vector<PointSpace> all_widths;
		all_widths.reserve (sum_of_region_sizes);
		for (RegionId r = 0; r < intervals.nb_regions (); ++r) {
			const auto & region_intervals = intervals.data (m, r);
			for (const auto & interval : region_intervals) {
				all_widths.emplace_back (interval.width);
			}
		}
		assert (all_widths.size () == sum_of_region_sizes);
		// Compute the median in the naive way. In C++17 std::nth_element would be a better O(n) solution.
		std::sort (all_widths.begin (), all_widths.end ());
		assert (sum_of_region_sizes > 0);
		if (sum_of_region_sizes % 2 == 1) {
			const auto mid_point_index = sum_of_region_sizes / 2;
			return all_widths[mid_point_index];
		} else {
			const auto above_mid_point_index = sum_of_region_sizes / 2;
			assert (above_mid_point_index > 0);
			return (all_widths[above_mid_point_index - 1] + all_widths[above_mid_point_index]) / 2.;
		}
	};

	std::vector<PointSpace> medians (intervals.nb_processes ());
	for (ProcessId m = 0; m < intervals.nb_processes (); ++m) {
		medians[m] = compute_median_width (m);
	}
	return medians;
}

int main (int argc, char * argv[]) {
	bool verbose = false;
	double gamma = 1.;

	BaseOption base_option = None{}; // Base choice starts undefined and MUST be defined
	KernelConfig kernel_config = KernelConfig::None;
	KernelType kernel_type = KernelType::Interval;
	Optional<std::vector<PointSpace>> override_homogeneous_kernel_widths;

	std::vector<string_view> current_region_names;
	std::vector<ProcessFile> process_files;

	// Command line parsing setup
	const auto command_line = CommandLineView (argc, argv);
	auto parser = CommandLineParser ();

	// Common
	parser.flag ({"h", "help"}, "Display this help", [&]() { //
		parser.usage (stderr, command_line.program_name ());
		std::exit (EXIT_SUCCESS);
	});
	parser.flag ({"v", "verbose"}, "Enable verbose output", [&]() { verbose = true; });

	// Hyper-parameters
	parser.option ({"g", "gamma"}, "value", "Set gamma value (double, positive)", [&gamma](string_view value) { //
		gamma = parse_strict_positive_double (value, "gamma");
	});

	// Base
	parser.option2 ({"histogram"}, "K", "delta", "Use an histogram base (k > 0, delta > 0)",
	                [&base_option](string_view k_value, string_view delta_value) {
		                const auto base_size = size_t (parse_strict_positive_int (k_value, "histogram K"));
		                const auto delta = PointSpace (parse_strict_positive_double (delta_value, "histogram delta"));
		                base_option = HistogramBase{base_size, delta};
	                });

	// Kernel setup
	parser.option ({"k", "kernel"}, "none|homogeneous|heterogeneous", "Kernel configuration (default=none)",
	               [&kernel_config](string_view value) {
		               if (value == "none") {
			               kernel_config = KernelConfig::None;
		               } else if (value == "homogeneous") {
			               kernel_config = KernelConfig::Homogeneous;
		               } else if (value == "heterogeneous") {
			               kernel_config = KernelConfig::Heterogeneous;
		               } else {
			               throw std::runtime_error (fmt::format ("Unknown kernel configuration: '{}'", value));
		               }
	               });
	parser.option ({"t", "kernel-type"}, "interval|interval_right_half", "Kernel type",
	               [&kernel_type](string_view value) {
		               if (value == "interval") {
			               kernel_type = KernelType::Interval;
		               } else if (value == "interval_right_half") {
			               kernel_type = KernelType::IntervalRightHalf;
		               } else {
			               throw std::runtime_error (fmt::format ("Unknown kernel type: '{}'", value));
		               }
	               });
	parser.option ({"homogeneous-kernel-widths"}, "w0[:w1:w2:...]", "Override kernel widths (default=median)",
	               [&override_homogeneous_kernel_widths](string_view values) {
		               override_homogeneous_kernel_widths = map_to_vector (split (':', values), [](string_view value) {
			               return PointSpace (parse_strict_positive_double (value, "kernel width"));
		               });
	               });

	// File parsing : generate a list of (file, options), read later.
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
	auto add_process_file = [&current_region_names, &process_files](string_view filename, ProcessDirection direction) {
		if (current_region_names.empty ()) {
			throw std::runtime_error ("List of region names is empty: set region names before reading a file");
		}
		process_files.emplace_back (ProcessFile{filename, current_region_names, direction});
	};
	parser.option ({"f", "file-forward"}, "filename", "Add process regions from file",
	               [add_process_file](string_view filename) { add_process_file (filename, ProcessDirection::Forward); });
	parser.option ({"b", "file-backward"}, "filename", "Add process regions (reversed) from file",
	               [add_process_file](string_view filename) { add_process_file (filename, ProcessDirection::Backward); });

	try {
		// Parse command line arguments. All actions declared to the parser will be called here.
		parser.parse (command_line);

		// Check that base is set
		struct CheckBase {
			BaseType operator() (const None &) const { throw std::runtime_error ("Function base is not defined"); }
			BaseType operator() (const HistogramBase & base) const { return base; }
		};
		const BaseType base = visit (CheckBase{}, base_option);

		// Read input files
		const auto intervals = read_process_files (process_files);

		// Post processing: determine kernel setup, generate sorted points lists
		const auto post_processing_start = instant ();
		const auto points = extract_point_lists (intervals);
		const auto kernels = [&]() -> variant<None, std::vector<IntervalKernel>> {
			// Helper: choose the source of kernel widths (explicit or deduced).
			// widths must be strictly positive.
			auto get_kernel_widths = [&]() -> std::vector<PointSpace> {
				if (override_homogeneous_kernel_widths) {
					if (override_homogeneous_kernel_widths.value.size () != process_files.size ()) {
						throw std::runtime_error (
						    fmt::format ("Explicit kernel widths number does not match number of processes: expected {}, got {}",
						                 process_files.size (), override_homogeneous_kernel_widths.value.size ()));
					}
					return std::move (override_homogeneous_kernel_widths.value);
				} else {
					auto widths = median_interval_widths (intervals);
					for (auto & w : widths) {
						w = std::max (w, 1.); // FIXME cannot be zero, what should be a minimum ??
					}
					fmt::print (stderr, "Using deduced kernel widths: {}\n", fmt::join (widths, ", "));
					return widths;
				}
			};

			if (kernel_type == KernelType::Interval) {
				return map_to_vector (get_kernel_widths (),
				                      [](PointSpace width) -> IntervalKernel { return IntervalKernel{width}; });
			} else {
				return None{};
			}
		}();
		const auto post_processing_end = instant ();
		fmt::print (stderr, "Post processing done: time = {}\n",
		            duration_string (post_processing_end - post_processing_start));

		// Compute base/kernel specific values: B, G, B_hat
		const auto compute_b_g_start = instant ();
		const auto intermediate_values = visit (
		    [&points](const auto & base, const auto & kernels) {
			    // Code to compute intermediate values for each combination of base and kernels.
			    return compute_intermediate_values (points, base, kernels);
		    },
		    base, kernels);
		const auto compute_b_g_end = instant ();
		fmt::print (stderr, "Computing B and G matrice done: time = {}\n",
		            duration_string (compute_b_g_end - compute_b_g_start));

		// Perform lassoshooting
		const auto lasso_start = instant ();
		const auto lasso_parameters = compute_lasso_parameters (intermediate_values, gamma);
		const auto estimated_a = compute_estimated_a_with_lasso (lasso_parameters);
		const auto lasso_end = instant ();
		fmt::print (stderr, "Lassoshooting done: time = {}\n", duration_string (lasso_end - lasso_start));

		// Print results
		if (verbose) {
			// Header
			fmt::print ("# Processes = {{\n");
			for (ProcessId m = 0; m < process_files.size (); ++m) {
				const auto & p = process_files[m];
				const string_view suffix = p.direction == ProcessDirection::Backward ? " (backward)" : "";
				fmt::print ("#  [{}] {}{}\n", m, p.filename, suffix);
			}
			fmt::print ("# }}\n");

			struct PrintBaseLine {
				void operator() (HistogramBase b) const {
					fmt::print ("# base = Histogram(K = {}, delta = {})\n", b.base_size, b.delta);
				}
			};
			visit (PrintBaseLine{}, base);

			struct PrintKernelLine {
				void operator() (None) const { fmt::print ("# kernels = None\n"); }
				void operator() (const std::vector<IntervalKernel> & kernels) const {
					const auto widths = map_to_vector (kernels, [](IntervalKernel k) { return k.width; });
					fmt::print ("# kernels = Intervals{{{}}}\n", fmt::join (widths, ", "));
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
