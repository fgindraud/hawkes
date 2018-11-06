#pragma once

#include <cassert>
#include <cstdint>
#include <eigen3/Eigen/Core>
#include <vector>

/******************************************************************************
 * Index types.
 */
struct ProcessId {
	int value; // [0; nb_processes[
};
struct FunctionBaseId {
	int value; // [0; base_size[
};

/******************************************************************************
 * Process data.
 */

// Single coordinate for a process represented by points.
using Point = std::int32_t;

template <typename T> class SortedVec {
private:
	std::vector<T> inner;

public:
	// TODO add constructors

	int size () const { return static_cast<int> (inner.size ()); }

	const T & operator[] (int i) const {
		assert (0 <= i && i < size ());
		return inner[static_cast<std::size_t> (i)];
	}

	using const_iterator = typename std::vector<T>::const_iterator;
	const_iterator begin () const { return inner.begin (); }
	const_iterator end () const { return inner.end (); }
};

struct SortedProcesses {
	std::vector<SortedVec<Point>> processes;

	int nb_processes () const { return int(processes.size ()); }

	const SortedVec<Point> & process (ProcessId l) const {
		assert (0 <= l.value && l.value < nb_processes ());
		return processes[std::size_t (l.value)];
	}
};

/******************************************************************************
 * Function bases.
 */
struct HistogramBase {
	int base_size; // [1, inf[
	int delta;     // [1, inf[

	// ]from; to]
	struct Interval {
		int from;
		int to;
	};

	Interval interval (FunctionBaseId k) const {
		assert (0 <= k.value && k.value < base_size);
		return {k.value * delta, (k.value + 1) * delta};
	}
};

/******************************************************************************
 * Computation matrices.
 */
struct MatrixB {
	// Stores values for b_m and estimated a_m for all m.
	// Invariant: K > 0 && M > 0 && inner.size() == (1 + K * M, M).

	int nb_processes; // M
	int base_size;    // K
	Eigen::MatrixXd inner;

	MatrixB (int nb_processes, int base_size)
	    : nb_processes (nb_processes), base_size (base_size), inner (1 + base_size * nb_processes, nb_processes) {
		assert (nb_processes > 0);
		assert (base_size > 0);
	}

	// b_m,0
	double get_0 (ProcessId m) const {
		assert (0 <= m.value && m.value < nb_processes);
		return inner (0, m.value);
	}
	void set_0 (ProcessId m, double v) {
		assert (0 <= m.value && m.value < nb_processes);
		inner (0, m.value) = v;
	}

	// b_m,l,k
	int lk_index (ProcessId l, FunctionBaseId k) const {
		assert (0 <= l.value && l.value < nb_processes);
		assert (0 <= k.value && k.value < base_size);
		return 1 + l.value * base_size + k.value;
	}
	double get_lk (ProcessId m, ProcessId l, FunctionBaseId k) const {
		assert (0 <= m.value && m.value < nb_processes);
		return inner (lk_index (l, k), m.value);
	}
	void set_lk (ProcessId m, ProcessId l, FunctionBaseId k, double v) {
		assert (0 <= m.value && m.value < nb_processes);
		inner (lk_index (l, k), m.value) = v;
	}
};
using MatrixA = MatrixB;

struct MatrixG {
	// Stores value of the G matrix (symmetric).
	// Invariant: K > 0 && M > 0 && inner.size() == (1 + K * M, 1 + K * M)

	int nb_processes; // M
	int base_size;    // K
	Eigen::MatrixXd inner;

	MatrixG (int nb_processes, int base_size)
	    : nb_processes (nb_processes),
	      base_size (base_size),
	      inner (1 + base_size * nb_processes, 1 + base_size * nb_processes) {
		assert (nb_processes > 0);
		assert (base_size > 0);
	}

	int lk_index (ProcessId l, FunctionBaseId k) const {
		assert (0 <= l.value && l.value < nb_processes);
		assert (0 <= k.value && k.value < base_size);
		return 1 + l.value * base_size + k.value;
	}

	// Tmax
	double get_tmax () const { return inner (0, 0); }
	void set_tmax (double v) { inner (0, 0) = v; }

	// g_l,k (duplicated with transposition)
	double get_g (ProcessId l, FunctionBaseId k) const { return inner (0, lk_index (l, k)); }
	void set_g (ProcessId l, FunctionBaseId k, double v) {
		const auto i = lk_index (l, k);
		inner (0, i) = inner (i, 0) = v;
	}

	// G_l,l2_k,k2 (symmetric, only need to be set for (l,k) <= [or >=] (l2,k2))
	double get_G (ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) const {
		return inner (lk_index (l, k), lk_index (l2, k2));
	}
	void set_G (ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2, double v) {
		const auto i = lk_index (l, k);
		const auto i2 = lk_index (l2, k2);
		inner (i, i2) = inner (i2, i) = v;
	}
};
