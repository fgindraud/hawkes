#pragma once

#include <cassert>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <vector>

struct ProcessId {
	int value; // [0; nb_processes[
};
struct FunctionBaseId {
	int value; // [0; base_size[
};

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

struct SortedProcess {
	std::vector<std::int32_t> points;

	int nb_points () const { return int(points.size ()); }

	std::int32_t point (int i) const {
		assert (0 <= i && i < nb_points ());
		return points[i];
	}
};
struct SortedProcesses {
	std::vector<SortedProcess> processes;

	int nb_processes () const { return int(processes.size ()); }

	const SortedProcess & process (ProcessId l) const {
		assert (0 <= l.value && l.value < nb_processes ());
		return processes[std::size_t (l.value)];
	}
};

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

inline int count_point_difference_in_interval (const SortedProcess & m_process, const SortedProcess & l_process,
                                               HistogramBase::Interval interval) {
	// Count pair of points within the k-th interval
	// TODO dichotomic search & count += interval_size instead of looking at each element
	// TODO impl by doing all k at once ? bench ?
	const auto n_m = m_process.nb_points ();
	int count = 0;
	int last_starting_i = 0;
	for (const auto x_l : l_process.points) {
		// Count x_m in ]x_l + interval.from, x_l + interval.to]

		// Find first i where Nm[i] is in the interval
		while (last_starting_i < n_m && !(x_l + interval.from < m_process.point (last_starting_i))) {
			last_starting_i += 1;
		}
		// i is out of bounds or Nm[i] > x_l + interval.from
		int i = last_starting_i;
		// Count elements of Nm still in interval
		while (i < n_m && m_process.point (i) <= x_l + interval.to) {
			count += 1;
			i += 1;
		}
	}
	return count;
}

inline int compute_cross_correlation (const SortedProcess & l_process, const SortedProcess & l2_process,
                                      HistogramBase::Interval interval, HistogramBase::Interval interval2) {
	return 0;
}

inline MatrixB compute_b (const SortedProcesses & processes, const HistogramBase & base) {
	const auto nb_processes = processes.nb_processes ();
	const auto base_size = base.base_size;
	MatrixB b (nb_processes, base_size);

	for (ProcessId m{0}; m.value < nb_processes; ++m.value) {
		const auto & m_process = processes.process (m);

		// b0
		b.set_0 (m, double(m_process.nb_points ()));

		// b_lk
		for (ProcessId l{0}; l.value < nb_processes; ++l.value) {
			for (FunctionBaseId k{0}; k.value < base_size; ++k.value) {
				const auto count = count_point_difference_in_interval (m_process, processes.process (l), base.interval (k));
				b.set_lk (m, l, k, double(count));
			}
		}
	}
	return b;
}

inline MatrixG compute_g (const SortedProcesses & processes, const HistogramBase & base) {
	const auto nb_processes = processes.nb_processes ();
	const auto base_size = base.base_size;
	MatrixG g (nb_processes, base_size);

	g.set_tmax (0); // FIXME how is Tmax determined ?

	for (ProcessId l{0}; l.value < nb_processes; ++l.value) {
		const auto g_lk = processes.process (l).nb_points () * std::sqrt (base.delta);
		for (FunctionBaseId k{0}; k.value < base_size; ++k.value) {
			g.set_g (l, k, g_lk);
		}
	}

	// G symmetric, only compute for (l2,k2) >= (l,k)
	for (ProcessId l{0}; l.value < nb_processes; ++l.value) {
		for (ProcessId l2{l.value}; l2.value < nb_processes; ++l2.value) {
			for (FunctionBaseId k{0}; k.value < base_size; ++k.value) {
				for (FunctionBaseId k2{k.value}; k2.value < base_size; ++k2.value) {
					const auto v = compute_cross_correlation (processes.process (l), processes.process (l2), base.interval (k),
					                                          base.interval (k2));
					g.set_G (l, l2, k, k2, double(v));
				}
			}
		}
	}
	return g;
}
