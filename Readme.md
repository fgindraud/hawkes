Hawkes
============

Estimate parameters for a hawkes model over spatial positions.

For multiple sets of points called _processes_, which each point being a spatial position (for example, mark on a chromosome).
Assume they have been generated by a hawkes process, modelised with a function base _phi(k)_.
Then estimate the weights of the hawkes process from actual points.

Usage
-----

Hawkes must be compiled before use, see the *How to build* section.
There are no runtime library dependencies.

Use `./hawkes -h` to access the list of command line options with a small description.

Example of use: `./hawkes -f process_a.bed -f process_b.bed -histogram 10 10000 -kernel homogeneous > output_file`.
- `-f filename` loads the process from _filename_. Use once for each process to be loaded.
- `-b filename` loads the process but reverses the dimension (`x -> -x` on the positions).
- `-histogram K delta` specifies the function base. Required.
- `-kernel config` what kernel configuration to use, defaults to none if not specified.
- `-kernel-type type` what kernel type to use (kernel function/shape), defaults to a centered interval if not specified.

The matrix of estimated weights is printed on the _standard output_, in a tab-separated text format.
Redirect the standard output using `<cmd> > file` to store the output in a file.
General information and progress is printed on the _standard error output_, and will not be redirected.
The _verbose_ mode adds a header to the output with the parameters used to generate the weights (process files, base, kernel setup).
This header is prefixed with `#` so that programs like _R_ can ignore it and load the matrix without problems.

The computations is splitted along _regions_, which represent independent spaces where points can exists.
For each region, only points of each process belonging to the region will be considered in the computation step.
Results are summed between regions afterwards, before the final LASSO step.
The region of a point is determined by the text in the first column of bed files:
	# region start end ; example with 2 regions (chr1, chr2)
	chr1 1 42
	chr1 78 102
	chr2 0 34
This program requires all points of a region to be in sequence (no interleaving of regions).
Regions missing in some of the processes will be cosidered to be empty regions.

See the BRP18 paper for the global model.
See the pdf documentation in `doc/shapes` for computation details.

How to build
------------

A C++ compiler supporting C++14 is required:
- g++ >= 5.0
- clang++ >= 3.4
- for any other compiler, check that it supports the `-std=c++14` flag

Simple compilation:
- `make` will compile the program with the default C++ compiler of your distribution, downloading a copy of Eigen (matrix library).
- `make CXX=<name_or_path_to_compiler_binary>` will use the specified compiler if the default is not valid.
- `make EIGEN_INCLUDE_PATH=/path/to/eigen/include` will use the indicated Eigen files instead of downloading a copy. Use if Eigen is already installed.

Advanced compilation:
- `make CXX_FLAGS_EXTRA='-static'` to generate a self-contained compiled version of Hawkes which should work on any machine with the same architecture and operating system.

Developer corner
----------------

`make hawkes_debug` will compile the program in debug mode, with some assertions enabled and debug informations generated.
This can be useful to investigate a bug with gdb, or print intermediate values (B/G matrices).

`make test` will compile and run a file with a lot of small tests of parts of the code.

`make format` will try to format the format the source code using the `clang-format` tool.

Hawkes uses 3 header-only libraries which are included in the current code:
- [fmtlib](http://fmtlib.net) formatting library to generate output text
- [mpark::variant](https://github.com/mpark/variant) to provide the C++17 variant sum type
- [doctest](https://github.com/onqtam/doctest) as a test framework for unit tests

License
-------

TODO
