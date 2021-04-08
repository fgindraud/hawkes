# Compilation flags.
# The C++14 flag in particular is required.
# Other useful flags:
# '-static' for a statically compiled binary (for a pre-compiled release version).
# '-march=native' for extra performance but looses portability.
# CXX_FLAGS_EXTRA can be used to pass additional flags for a custom compilation without modifying the script.
CXX_FLAGS_COMMON = -std=c++14 -Wall -Wextra $(CXX_FLAGS_EXTRA)
CXX_FLAGS_RELEASE = $(CXX_FLAGS_COMMON) -O3 -DNDEBUG
CXX_FLAGS_DEBUG = $(CXX_FLAGS_COMMON) -O2 -g

.PHONY: all clean test format

# Default target.
# Explicitely call 'make hawkes_debug' for a version with debug info (for use with gdb / valgrind)
all: hawkes

# Eigen C++ library configuration
# Define EIGEN_INCLUDE_PATH as location of eigen include files AND as a target representing a dependency to eigen code.
# EIGEN_INCLUDE_PATH can be set to the location of an already "installed" eigen copy (for example /usr/include/eigen3 for the system one, this might be different depending on the distribution).
ifndef EIGEN_INCLUDE_PATH
# If not set, a local copy of eigen is unpacked from a tar.gz and used.
EIGEN_INCLUDE_PATH := eigen-3.3.9
$(EIGEN_INCLUDE_PATH): $(EIGEN_INCLUDE_PATH).tar.gz
	tar xf $<
	touch -c $@
endif
CXX_FLAGS_COMMON += -I $(EIGEN_INCLUDE_PATH)

# Precompile fmtlib to reduce compilation time
fmtlib.o: src/external/fmt/format.cc $(wildcard src/external/fmt/*.h) Makefile
	$(CXX) -c $(CXX_FLAGS_RELEASE) -I src/external -o $@ $<

# Debug and non-debug version of hawkes main program.
hawkes: src/main.cpp fmtlib.o $(EIGEN_INCLUDE_PATH) $(wildcard src/*.h) Makefile
	$(CXX) $(CXX_FLAGS_RELEASE) -o $@ $< fmtlib.o
hawkes_debug: src/main.cpp fmtlib.o $(EIGEN_INCLUDE_PATH) $(wildcard src/*.h) Makefile
	$(CXX) $(CXX_FLAGS_DEBUG) -o $@ $< fmtlib.o

# Unit test executable and 'test' target.
unit_tests: src/unit_tests.cpp fmtlib.o $(EIGEN_INCLUDE_PATH) $(wildcard src/*.h) Makefile
	$(CXX) $(CXX_FLAGS_DEBUG) -o $@ $< fmtlib.o
test: unit_tests
	./unit_tests

# Small utility to plot various shapes, for debug purposes.
dump_shape: src/dump_shape.cpp fmtlib.o $(EIGEN_INCLUDE_PATH) $(wildcard src/*.h) Makefile
	$(CXX) $(CXX_FLAGS_DEBUG) -o $@ $< fmtlib.o

# Utility used to compute part of a goodness statistics.
goodness: src/goodness.cpp fmtlib.o $(EIGEN_INCLUDE_PATH) $(wildcard src/*.h) Makefile
	$(CXX) $(CXX_FLAGS_DEBUG) -o $@ $< fmtlib.o

clean:
	$(RM) fmtlib.o
	$(RM) hawkes
	$(RM) hawkes_debug
	$(RM) unit_tests
	$(RM) dump_shape
	$(RM) goodness

format:
	clang-format -style=file -i -verbose src/main.cpp src/unit_tests.cpp src/dump_shape.cpp src/goodness.cpp $(wildcard src/*.h)
