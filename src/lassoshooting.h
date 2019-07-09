#pragma once

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "utils.h"

/* For x,b vectors of size n, G matrix of size (n,n):
 * result = argmin_{x} (-2*t(x)*b + t(x)*G*x + lambda*t(weights)*|x|).
 * Code is a rewrite of the lassoshooting R module C source code, using Eigen.
 */
inline Eigen::VectorXd lassoshooting (const Eigen::MatrixXd & xtx, Eigen::VectorXd xty,
                                      const Eigen::VectorXd & penaltyweights, const double lambda) {
	// xtx is matrix G
	// xty is vector b
	const auto p = xty.size ();
	assert (p == penaltyweights.size ());
	assert (xtx.cols () == p && xtx.rows () == p);

	// soft_threshold utility function
	const auto soft_threshold = [](double x, double t) {
		const auto v = std::abs (x) - t;
		if (v > 0.) {
			if (x >= 0.) {
				return v;
			} else {
				return -v;
			}
		} else {
			return 0.;
		}
	};

	/* From initial code:
	 * factor2 = 1.
	 * nopenalize = nullptr
	 * params->w = penaltyweights (set)
	 * params->XtX = xtx
	 * params->Xty = xty
	 * params->s = nullptr
	 * params->forcezero = -1
	 *
	 * daxpy(N, a, *x, incx, *y, iny) : y[i] += a * x[i], with incx/incy strides, N total size.
	 */
	const double convergence_threshold = 1E-6;
	const int max_iteration_count = 10000;

	/* params->beta are the output coefficients.
	 * They are also an optional starting point for the iterative algorithm.
	 * In our case, we do not use a starting point, so they default to all zeroes.
	 */
	Eigen::VectorXd beta = Eigen::VectorXd::Zero (p);

	const auto xty_infinity_norm = xty.array ().abs ().maxCoeff ();
	if (lambda > xty_infinity_norm) {
		throw std::runtime_error ("lasso: lambda > xty_infinity_norm");
	}

	auto & s = xty;                  // beta is initially unset so compensation is a daxpy(alpha=0) == noop.
	const auto & w = penaltyweights; // nopenalize is unset, no modifications of w

	for (int nb_iteration = 0; nb_iteration < max_iteration_count; ++nb_iteration) {
		double delta = 0;

		for (int j = 0; j < p; ++j) {
			const double xtx_jj = xtx (j, j);
			if (xtx_jj == 0.) {
				continue;
			}

			const double tmp = s[j] + xtx_jj * beta[j];
			if (std::isnan (tmp) || std::isinf (tmp)) {
#ifndef NDEBUG
				fmt::print (stderr, "Lasso: j = {} ; s[j] = {} ; beta[j] = {} ; xtx[j,j] = {}\n", j, s[j], beta[j], xtx_jj);
#endif
				throw std::runtime_error (fmt::format ("lasso: bad beta_j_star: {} ; iteration={}", tmp, nb_iteration));
			}

			const double new_beta_j = soft_threshold (tmp, w[j] * lambda) / xtx_jj;
			const double delta_beta_j = new_beta_j - beta[j];
			delta = std::max (delta, std::abs (delta_beta_j));
			beta[j] = new_beta_j;

			s += (-delta_beta_j) * xtx.col (j);
		}

		if (delta <= convergence_threshold) {
			break;
		}
	}
	return beta;
}

inline Eigen::VectorXd lassoshooting_g (const Eigen::MatrixXd & g, const Eigen::VectorXd & b, Eigen::VectorXd d,
                                        const double lambda) {
	const auto p = b.size ();
	assert (p == d.size ());
	assert (g.cols () == p && g.rows () == p);

	d *= lambda;

	const auto minimized_expression = [&g, &b, &d](const Eigen::VectorXd & a) -> double {
		return 0.5 * double(a.transpose () * g * a) - double(b.transpose () * a) + double(d.transpose () * a.cwiseAbs ());
	};

	const int max_iteration_count = 10000;
	const double convergence_threshold = 1E-6;

	Eigen::VectorXd a = Eigen::VectorXd::Zero (p);
	for (int iteration = 0; iteration < max_iteration_count; ++iteration) {
		const double previous_expression_value = minimized_expression (a);
		for (int i = 0; i < p; ++i) {
			const auto s = double(g.row (i) * a) - g (i, i) * a (i) - b (i);
			if ((-s - d (i)) > 0) {
				a (i) = (-s - d (i)) / g (i, i);
			} else if ((-s + d (i)) < 0) {
				a (i) = (-s + d (i)) / g (i, i);
			} else {
				a (i) = 0;
			}
		}
		if (std::abs (minimized_expression (a) - previous_expression_value) < convergence_threshold) {
			break;
		}
	}
	return a;
}
