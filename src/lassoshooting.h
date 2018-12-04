#pragma once

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <stdexcept>

/* For x,b vectors of size n, G matrix of size (n,n):
 * result = argmin_{x} (-2*t(x)*b + t(x)*G*x + lambda*t(weights)*|x|).
 * Code is a rewrite of the lassoshooting R module C source code, using Eigen.
 */
inline Eigen::VectorXd lassoshooting (const Eigen::MatrixXd & xtx, const Eigen::VectorXd & xty,
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

	Eigen::VectorXd s = xty;         // beta is initially unset so compensation is a daxpy(alpha=0) == noop.
	const auto & w = penaltyweights; // nopenalize is unset, no modifications of w

	for (int nb_iteration = 0; nb_iteration < max_iteration_count; ++nb_iteration) {
		double delta = 0;

		for (int j = 0; j < p; ++j) {
			const auto xtx_jj = xtx (j, j);
			if (xtx_jj == 0.) {
				continue;
			}

			const auto beta_j_star = s[j] + xtx_jj * beta[j];
			if (std::isnan (beta_j_star) || std::isinf (beta_j_star)) {
				throw std::runtime_error ("lasso: bad beta_j_star");
			}

			const auto new_beta_j = soft_threshold (beta_j_star, w[j] * lambda) / xtx_jj;
			const auto delta_beta_j = new_beta_j - beta[j];
			delta = std::max (delta, std::abs (delta_beta_j));
			beta[j] = new_beta_j;

			s -= -delta_beta_j * xtx.col (j);
		}

		if (delta <= convergence_threshold) {
			break;
		}
	}
	return beta;
}
