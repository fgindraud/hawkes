# Compilation flags.
# The C++14 flag in particular is required.
# Other useful flags:
# '-static' for a statically compiled binary (for a pre-compiled release version).
# '-march=native' for extra performance but looses portability.
# CXX_FLAGS_EXTRA can be used to pass additional flags for a custom compilation without modifying the script.
CXX_FLAGS_COMMON = -std=c++14 -Wall -Wextra $(CXX_FLAGS_EXTRA)
CXX_FLAGS_RELEASE = $(CXX_FLAGS_COMMON) -O3 -DNDEBUG
CXX_FLAGS_DEBUG = $(CXX_FLAGS_COMMON) -O2 -g

.PHONY: all clean test format eigen

# Default target.
# Explicitely call 'make hawkes_debug' for a version with debug info (for use with gdb / valgrind)
all: hawkes

# Eigen C++ library configuration
# Define an 'eigen' target representing a dependency to eigen code.
ifdef EIGEN_INCLUDE_PATH
# Using the system installed eigen, at the given path
eigen:
CXX_FLAGS_COMMON += -I $(EIGEN_INCLUDE_PATH)
else
# Download and uncompress eigen in a subdirectory, then use it.
# Version and directory names must be updated if the eigen version is changed!
EIGEN_TAR_DIR := eigen-eigen-323c052e1731

eigen.tar.gz:
	curl -L -o $@ "http://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz"
$(EIGEN_TAR_DIR): eigen.tar.gz
	tar xf $<
eigen: $(EIGEN_TAR_DIR)

CXX_FLAGS_COMMON += -I $(EIGEN_TAR_DIR)
endif

# Precompile fmtlib to reduce compilation time
fmtlib.o: src/external/fmt/format.cc $(wildcard src/external/fmt/*.h) Makefile
	$(CXX) -c $(CXX_FLAGS_RELEASE) -I src/external -o $@ $<

# Debug and non-debug version of hawkes main program.
hawkes: src/main.cpp fmtlib.o eigen $(wildcard src/*.h) Makefile
	$(CXX) $(CXX_FLAGS_RELEASE) -o $@ $< fmtlib.o
hawkes_debug: src/main.cpp fmtlib.o eigen $(wildcard src/*.h) Makefile
	$(CXX) $(CXX_FLAGS_DEBUG) -o $@ $< fmtlib.o

# Unit test executable and 'test' target.
unit_tests: src/unit_tests.cpp fmtlib.o eigen $(wildcard src/*.h) Makefile
	$(CXX) $(CXX_FLAGS_DEBUG) -o $@ $< fmtlib.o
test: unit_tests
	./unit_tests

# Small utility to plot various shapes, for debug purposes.
dump_shape: src/dump_shape.cpp fmtlib.o eigen $(wildcard src/*.h) Makefile
	$(CXX) $(CXX_FLAGS_DEBUG) -o $@ $< fmtlib.o

clean:
	$(RM) fmtlib.o
	$(RM) hawkes
	$(RM) hawkes_debug
	$(RM) unit_tests
	$(RM) dump_shape

format:
	clang-format -style=file -i -verbose src/main.cpp src/unit_tests.cpp src/dump_shape.cpp $(wildcard src/*.h)
