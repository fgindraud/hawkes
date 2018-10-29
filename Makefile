# Compilation flags.
# The C++17 flag in particular is required.
# Other useful flags:
# '-static' for a statically compiled binary (for a pre-compiled release version).
# '-march=native' for extra performance but looses portability.
# '-fopenmp' for OpenMP multithreading
# CXX_FLAGS_EXTRA can be used to pass additional flags for a custom compilation without modifying the script.
CXX_FLAGS_COMMON = -std=c++17 -Wall -Wextra $(CXX_FLAGS_EXTRA)
CXX_FLAGS_RELEASE = $(CXX_FLAGS_COMMON) -O3 -DNDEBUG
CXX_FLAGS_DEBUG = $(CXX_FLAGS_COMMON) -O2 -g

.PHONY: all clean test format

# Default target.
# Explicitely call 'make hawkes_debug' for a version with debug info (for use with gdb / valgrind)
all: hawkes

# Precompile fmtlib for speed
fmtlib.o: src/external/fmt/format.cc $(wildcard src/external/fmt/*.h) Makefile
	$(CXX) -c $(CXX_FLAGS_RELEASE) -I src/external -o $@ $<

# All possible binaries
hawkes: src/main.cpp fmtlib.o $(wildcard src/*.h) Makefile
	$(CXX) $(CXX_FLAGS_RELEASE) -o $@ $< fmtlib.o

hawkes_debug: src/main.cpp fmtlib.o $(wildcard src/*.h) Makefile
	$(CXX) $(CXX_FLAGS_DEBUG) -o $@ $< fmtlib.o

# FIXME restore tests
unit_tests: src/unit_tests.cpp $(wildcard src/*.h) Makefile
	$(CXX) $(CXX_FLAGS_DEBUG) -o $@ $<

test: unit_tests
	./unit_tests

clean:
	$(RM) hawkes
	$(RM) hawkes_debug
	$(RM) unit_tests

format:
	clang-format -style=file -i -verbose src/main.cpp src/unit_tests.cpp $(wildcard src/*.h)
