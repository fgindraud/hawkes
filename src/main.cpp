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
template <typename DataType>
static void read_process_data_from (ProcessesData<DataType> & processes, string_view filename,
                                    span<const string_view> region_names) {
	try {
		const auto start = instant ();
		auto file = open_file (filename, "r");
		auto data = read_selected_from_bed_file<DataType> (file.get (), region_names);
		const auto id = processes.add_process (filename, std::move (data));
		const auto end = instant ();
		fmt::print (stderr, "Process {} loaded from {}: regions = {} ; time = {}\n", id.value, filename,
		            processes.nb_regions (), duration_string (end - start));
	} catch (const std::runtime_error & e) {
		throw std::runtime_error (fmt::format ("Reading process data from {}: {}", filename, e.what ()));
	}
}

/******************************************************************************
 * Tests
 */
#include <iostream>
template <typename DataType> static void do_test (const ProcessesData<DataType> & processes) {
	for (int delta = 10; delta < 1000000; delta *= 10) {
		fmt::print ("### Delta = {}\n", delta);
		HistogramBase base{7, delta};
		std::vector<Matrix_M_MK1> matrix_b;
		std::vector<MatrixG> matrix_g;
		{
			const auto start = instant ();
			for (RegionId r{0}; r.value < processes.nb_regions (); ++r.value) {
				matrix_b.emplace_back (compute_b (processes, r, base));
			}
			const auto end = instant ();
			fmt::print (stderr, "matrix_b: time = {}\n", duration_string (end - start));
		}
		{
			const auto start = instant ();
			for (RegionId r{0}; r.value < processes.nb_regions (); ++r.value) {
				matrix_g.emplace_back (compute_g (processes, r, base));
			}
			const auto end = instant ();
			fmt::print (stderr, "matrix_g: time = {}\n", duration_string (end - start));
		}
		std::cerr << matrix_g[0].inner << "\n";
	}
}

/******************************************************************************
 * Program entry point.
 */
int main (int argc, char * argv[]) {
	ProcessesData<Point> point_processes;

	// Command line parsing setup
	const auto command_line = CommandLineView (argc, argv);
	auto parser = CommandLineParser ();

	parser.flag ({"h", "help"}, "Display this help", [&]() { //
		parser.usage (stderr, command_line.program_name ());
		std::exit (EXIT_SUCCESS);
	});

#if defined(_OPENMP)
	parser.option ({"n", "nb-threads"}, "n", "Number of computation threads (default=max)", [&](string_view n) {
		const auto nb_threads = parse_positive_int (n, "nb threads");
		const auto nb_proc = std::size_t (omp_get_num_procs ());
		if (!(0 < nb_threads && nb_threads <= nb_proc)) {
			throw std::runtime_error (fmt::format ("Number of threads must be in [1, {}]", nb_proc));
		}
		omp_set_num_threads (int(nb_threads));
	});
#endif

	parser.option2 ({"f"}, "filename", "r1[,r2,...]", "Add selected regions from process file",
	                [&](string_view filename, string_view regions) {
		                auto region_names = split (',', regions);
		                read_process_data_from (point_processes, filename, make_span (region_names));
	                });

	try {
		// Parse command line arguments. All actions declared to the parser will be called here.
		parser.parse (command_line);

		// do_test (point_processes);
		const auto base = HistogramBase{4, 10};
		point_processes.add_process ("p1", {{"r1", SortedVec<Point>::from_sorted ({5, 15})}});
		point_processes.add_process ("p2", {{"r1", SortedVec<Point>::from_sorted ({6, 18})}});
		const std::vector<IntervalKernel> kernels = {IntervalKernel{6}, IntervalKernel{6}};
		const auto g_points = compute_g (point_processes, RegionId{0}, base);
		const auto g_kernels = compute_g (point_processes, RegionId{0}, base, make_span (kernels));
		std::cerr << "=== G with points ===\n" << g_points.inner << "\n";
		std::cerr << "=== G with kernels ===\n" << g_kernels.inner << "\n";

	} catch (const CommandLineParser::Exception & exc) {
		fmt::print (stderr, "Error: {}. Use --help for a list of options.\n", exc.what ());
		return EXIT_FAILURE;
	} catch (const std::exception & exc) {
		fmt::print (stderr, "Error: {}\n", exc.what ());
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
