#pragma once

#include <Eigen/Cholesky>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "lassoshooting.h"
#include "shape.h"
#include "types.h"

/* Compute Tmax (used in G).
 * Tmax is the maximum width covered by points of all processes for a specific region.
 */
inline PointSpace tmax(span<const SortedVec<Point>> processes) {
    PointSpace min = std::numeric_limits<PointSpace>::max();
    PointSpace max = std::numeric_limits<PointSpace>::min();

    for(const auto & points : processes) {
        if(points.size() > 0) {
            min = std::min(min, points[0]);
            max = std::max(max, points[points.size() - 1]);
        }
    }

    if(min <= max) {
        return max - min;
    } else {
        return 0.; // If there are no points at all, return 0
    }
}

/* Conversion of objects to shapes.
 */

inline auto to_shape(IntervalKernel kernel) {
    return shape::scaled(
        normalization_factor(kernel),
        shape::Indicator<Bound::Closed, Bound::Closed>{
            {-kernel.width / 2., kernel.width / 2.},
        });
}
inline auto to_shape(const HistogramBase::Histogram & histogram) {
    return shape::scaled(
        histogram.normalization_factor, shape::Indicator<Bound::Open, Bound::Closed>{histogram.interval});
}

inline auto up_shape(const HaarBase::Wavelet & w) {
    return shape::scaled(w.normalization_factor, shape::Indicator<Bound::Open, Bound::Closed>{w.up_part});
}
inline auto down_shape(const HaarBase::Wavelet & w) {
    return shape::scaled(w.normalization_factor, shape::Indicator<Bound::Open, Bound::Closed>{w.down_part});
}

/******************************************************************************
 * Compute B, G, B_hat intermediate values.
 *
 * This file defines overloads of compute_intermediate_values for many cases.
 * Each case is defined by the type of arguments:
 * - function base,
 * - kernel configuration.
 * All cases returns the same type of values, but use the specific computation code for this case.
 * The right implementation is then selected depending on the actual types of base and kernel configuration.
 */

// Common struct storing computation results.
struct CommonIntermediateValues {
    std::vector<Matrix_M_MK1> b_by_region;
    std::vector<MatrixG> g_by_region;
    Matrix_M_MK1 b_hat;
};

// List of defined overloads for implemented cases
inline CommonIntermediateValues compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points, const HistogramBase & base);
inline CommonIntermediateValues compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points,
    const HistogramBase & base,
    const HomogeneousKernels<IntervalKernel> & kernels);
inline CommonIntermediateValues compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points,
    const HistogramBase & base,
    const HeterogeneousKernels<IntervalKernel> & kernels);
inline CommonIntermediateValues compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points, const HaarBase & base);

inline void print_supported_computation_cases() {
    fmt::print("histogram base with no kernel\n");
    fmt::print("histogram base with homogenous interval kernels\n");
    fmt::print("histogram base with heterogeneous interval kernels\n");
    fmt::print("haar wavelet base with no kernel\n");
}

// Overload on base classes : perform selection.
// This emulates pattern matching on sum types.
inline CommonIntermediateValues compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points, const Base & base, const KernelConfig & kernel_config) {
    // First select on base type, then kernel config
    if(auto * histogram = dynamic_cast<const HistogramBase *>(&base)) {
        if(dynamic_cast<const NoKernel *>(&kernel_config)) {
            return compute_intermediate_values(points, *histogram);
        } else if(auto * kernels = dynamic_cast<const HomogeneousKernels<IntervalKernel> *>(&kernel_config)) {
            return compute_intermediate_values(points, *histogram, *kernels);
        } else if(auto * kernels = dynamic_cast<const HeterogeneousKernels<IntervalKernel> *>(&kernel_config)) {
            return compute_intermediate_values(points, *histogram, *kernels);
        }
    } else if(auto * haar = dynamic_cast<const HaarBase *>(&base)) {
        if(dynamic_cast<const NoKernel *>(&kernel_config)) {
            return compute_intermediate_values(points, *haar);
        }
    }

    throw std::runtime_error(
        fmt::format("The combination of '{}' with '{}' is not supported", base.name(), kernel_config.name()));
}

/******************************************************************************
 * Basic histogram case, no kernel.
 */
inline Matrix_M_MK1 compute_b(span<const SortedVec<Point>> points, const HistogramBase & base) {
    const auto nb_processes = points.size();
    const auto base_size = base.base_size;
    Matrix_M_MK1 b(nb_processes, base_size);

    for(ProcessId m = 0; m < nb_processes; ++m) {
        // b_0
        b.set_0(m, double(points[m].size()));
        // b_lk
        for(ProcessId l = 0; l < nb_processes; ++l) {
            for(FunctionBaseId k = 0; k < base_size; ++k) {
                const double b_mlk = sum_of_point_differences(points[m], points[l], to_shape(base.histogram(k)));
                b.set_lk(m, l, k, b_mlk);
            }
        }
    }
    return b;
}

inline MatrixG compute_g(span<const SortedVec<Point>> points, const HistogramBase & base) {
    const auto nb_processes = points.size();
    const auto base_size = base.base_size;
    MatrixG g(nb_processes, base_size);

    g.set_tmax(tmax(points));

    const auto sqrt_delta = std::sqrt(base.delta);
    for(ProcessId l = 0; l < nb_processes; ++l) {
        const auto g_lk = double(points[l].size()) * sqrt_delta;
        for(FunctionBaseId k = 0; k < base_size; ++k) {
            g.set_g(l, k, g_lk);
        }
    }

    auto G_ll2kk2 = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
        const auto shape = cross_correlation(to_shape(base.histogram(k)), to_shape(base.histogram(k2)));
        return sum_of_point_differences(points[l], points[l2], shape);
    };
    /* G symmetric, only compute for (l2,k2) >= (l,k) (lexicographically).
     *
     * G_{l,l2,k,k2} = integral_x sum_{x_l,x_l2} phi_k (x - x_l) phi_k2 (x - x_l2) dx.
     * By change of variable x -> x + c*delta: G_{l,l2,k,k2} = G_{l,l2,k+c,k2+c}.
     * Thus for each block of G for (l,l2), we only need to compute 2K+1 values of G.
     * We compute the values on two borders, and copy the values for all inner cells.
     */
    for(ProcessId l = 0; l < nb_processes; ++l) {
        // Case l2 == l: compute for k2 >= k:
        for(FunctionBaseId k = 0; k < base_size; ++k) {
            // Compute G_{l,l,0,k} and copy to G_{l,l,c,k+c} for k in [0,K[.
            const auto v = G_ll2kk2(l, l, FunctionBaseId{0}, k);
            for(size_t c = 0; k + c < base_size; ++c) {
                g.set_G(l, l, FunctionBaseId{c}, FunctionBaseId{k + c}, v);
            }
        }
        // Case l2 > l: compute for all (k,k2):
        for(ProcessId l2 = l; l2 < nb_processes; ++l2) {
            // Compute G_{l,l2,0,0} and copy to G_{l,l2,c,c}.
            {
                const auto v = G_ll2kk2(l, l2, FunctionBaseId{0}, FunctionBaseId{0});
                for(size_t c = 0; c < base_size; ++c) {
                    g.set_G(l, l2, FunctionBaseId{c}, FunctionBaseId{c}, v);
                }
            }
            /* for k in [1,K[:
             * Compute G_{l,l2,0,k} and copy to G_{l,l2,c,k+c}.
             * Compute G_{l,l2,k,0} and copy to G_{l,l2,k+c,c}.
             */
            for(FunctionBaseId k = 1; k < base_size; ++k) {
                const auto v_0k = G_ll2kk2(l, l2, FunctionBaseId{0}, k);
                const auto v_k0 = G_ll2kk2(l, l2, k, FunctionBaseId{0});
                for(size_t c = 0; k + c < base_size; ++c) {
                    g.set_G(l, l2, FunctionBaseId{c}, FunctionBaseId{k + c}, v_0k);
                    g.set_G(l, l2, FunctionBaseId{k + c}, FunctionBaseId{c}, v_k0);
                }
            }
        }
    }
    return g;
}

// Complexity: O( M^2 * K + M * max(|N_m|) )
// Computation is exact (sup can be evaluated perfectly).
inline Matrix_M_MK1 compute_b_hat(const DataByProcessRegion<SortedVec<Point>> & points, const HistogramBase & base) {
    const auto nb_processes = points.nb_processes();
    const auto nb_regions = points.nb_regions();
    const auto base_size = base.base_size;
    Matrix_M_MK1 b_hat(nb_processes, base_size);

    // The sup is not affected by translation, so sup(phi_0) = sup(phi_k)
    // Only compute for phi_0 and replicate for all phi_k.
    const auto phi_0 = to_shape(base.histogram(FunctionBaseId{0}));

    for(ProcessId l = 0; l < nb_processes; ++l) {
        double b_hat_l = 0.;
        for(RegionId r = 0; r < nb_regions; ++r) {
            b_hat_l += sup_of_sum_of_differences_to_points(points.data(l, r), phi_0);
        }

        for(ProcessId m = 0; m < nb_processes; ++m) {
            for(FunctionBaseId k = 0; k < base_size; ++k) {
                b_hat.set_lk(m, l, k, b_hat_l);
            }
        }
    }
    return b_hat;
}

inline CommonIntermediateValues compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points, const HistogramBase & base) {
    const auto nb_regions = points.nb_regions();
    std::vector<Matrix_M_MK1> b_by_region;
    std::vector<MatrixG> g_by_region;
    b_by_region.reserve(nb_regions);
    g_by_region.reserve(nb_regions);
    for(RegionId r = 0; r < nb_regions; ++r) {
        b_by_region.emplace_back(compute_b(points.data_for_region(r), base));
        g_by_region.emplace_back(compute_g(points.data_for_region(r), base));
    }
    return {std::move(b_by_region), std::move(g_by_region), compute_b_hat(points, base)};
}

/******************************************************************************
 * Histogram with interval convolution kernels.
 *
 * In this case we use the shape strategy.
 * Values of B/G are expressed from convolution of shapes coming from the base and kernels.
 */
inline Matrix_M_MK1 compute_b(
    span<const SortedVec<Point>> points, const HistogramBase & base, const std::vector<IntervalKernel> & kernels) {
    assert(kernels.size() == points.size());
    const auto nb_processes = points.size();
    const auto base_size = base.base_size;
    Matrix_M_MK1 b(nb_processes, base_size);

    for(ProcessId m = 0; m < nb_processes; ++m) {
        // b0
        b.set_0(m, double(points[m].size()) * std::sqrt(kernels[m].width));
        // b_lk
        for(ProcessId l = 0; l < nb_processes; ++l) {
            for(FunctionBaseId k = 0; k < base_size; ++k) {
                const auto shape =
                    convolution(to_shape(base.histogram(k)), convolution(to_shape(kernels[m]), to_shape(kernels[l])));
                const double b_mlk = sum_of_point_differences(points[m], points[l], shape);
                b.set_lk(m, l, k, b_mlk);
            }
        }
    }
    return b;
}

inline MatrixG compute_g(
    span<const SortedVec<Point>> points, const HistogramBase & base, const std::vector<IntervalKernel> & kernels) {
    assert(kernels.size() == points.size());
    const auto nb_processes = points.size();
    const auto base_size = base.base_size;
    const auto sqrt_delta = std::sqrt(double(base.delta));
    MatrixG g(nb_processes, base_size);

    g.set_tmax(tmax(points));

    for(ProcessId l = 0; l < nb_processes; ++l) {
        /* g_lk = sum_{x_m} integral convolution(w_l,phi_k) (x - x_m) dx.
         * g_lk = sum_{x_m} (integral w_l) (integral phi_k) = sum_{x_m} eta_l sqrt(delta) = |N_m| eta_l sqrt(delta).
         */
        const auto g_lk = double(points[l].size()) * std::sqrt(kernels[l].width) * sqrt_delta;
        for(FunctionBaseId k = 0; k < base_size; ++k) {
            g.set_g(l, k, g_lk);
        }
    }

    const auto G_ll2kk2 = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
        // V = sum_{x_l,x_l2} corr(conv(W_l,phi_k),conv(W_l2,phi_k2)) (x_l-x_l2)
        const auto shape = cross_correlation(
            convolution(to_shape(kernels[l]), to_shape(base.histogram(k))),
            convolution(to_shape(kernels[l2]), to_shape(base.histogram(k2))));
        return sum_of_point_differences(points[l], points[l2], shape);
    };
    /* G symmetric, only compute for (l2,k2) >= (l,k) (lexicographically).
     *
     * G_{l,l2,k,k2} = integral_x sum_{x_l,x_l2} conv(w_l,phi_k) (x - x_l) conv(w_l2,phi_k2) (x - x_l2) dx.
     * By change of variable x -> x + c*delta: G_{l,l2,k,k2} = G_{l,l2,k+c,k2+c}.
     * Thus for each block of G for (l,l2), we only need to compute 2K+1 values of G.
     * We compute the values on two borders, and copy the values for all inner cells.
     */
    for(ProcessId l = 0; l < nb_processes; ++l) {
        // Case l2 == l: compute for k2 >= k:
        for(FunctionBaseId k = 0; k < base_size; ++k) {
            // Compute G_{l,l,0,k} and copy to G_{l,l,c,k+c} for k in [0,K[.
            const auto v = G_ll2kk2(l, l, FunctionBaseId{0}, k);
            for(size_t c = 0; k + c < base_size; ++c) {
                g.set_G(l, l, FunctionBaseId{c}, FunctionBaseId{k + c}, v);
            }
        }
        // Case l2 > l: compute for all (k,k2):
        for(ProcessId l2 = l; l2 < nb_processes; ++l2) {
            // Compute G_{l,l2,0,0} and copy to G_{l,l2,c,c}.
            {
                const auto v = G_ll2kk2(l, l2, FunctionBaseId{0}, FunctionBaseId{0});
                for(size_t c = 0; c < base_size; ++c) {
                    g.set_G(l, l2, FunctionBaseId{c}, FunctionBaseId{c}, v);
                }
            }
            /* for k in [1,K[:
             * Compute G_{l,l2,0,k} and copy to G_{l,l2,c,k+c}.
             * Compute G_{l,l2,k,0} and copy to G_{l,l2,k+c,c}.
             */
            for(FunctionBaseId k = 1; k < base_size; ++k) {
                const auto v_0k = G_ll2kk2(l, l2, FunctionBaseId{0}, k);
                const auto v_k0 = G_ll2kk2(l, l2, k, FunctionBaseId{0});
                for(size_t c = 0; k + c < base_size; ++c) {
                    g.set_G(l, l2, FunctionBaseId{c}, FunctionBaseId{k + c}, v_0k);
                    g.set_G(l, l2, FunctionBaseId{k + c}, FunctionBaseId{c}, v_k0);
                }
            }
        }
    }
    return g;
}

// Approximated
inline Matrix_M_MK1 compute_b_hat(
    const DataByProcessRegion<SortedVec<Point>> & points,
    const HistogramBase & base,
    const std::vector<IntervalKernel> & kernels) {
    assert(kernels.size() == points.nb_processes());
    const auto nb_processes = points.nb_processes();
    const auto nb_regions = points.nb_regions();
    const auto base_size = base.base_size;
    Matrix_M_MK1 b_hat(nb_processes, base_size);

    for(ProcessId l = 0; l < nb_processes; ++l) {
        for(ProcessId m = 0; m < nb_processes; ++m) {
            for(FunctionBaseId k = 0; k < base_size; ++k) {
                // Base shapes
                const auto w_m = to_shape(kernels[m]);
                const auto w_l = to_shape(kernels[l]);
                const auto phi_k = to_shape(base.histogram(k));
                // Approximate trapezoids with an interval of height=avg, width=width of trapezoid
                const auto approx = indicator_approximation(convolution(phi_k, convolution(w_m, w_l)));

                double sum_of_region_sups = 0;
                for(RegionId r = 0; r < nb_regions; ++r) {
                    sum_of_region_sups += sup_of_sum_of_differences_to_points(points.data(l, r), approx);
                }
                b_hat.set_lk(m, l, k, sum_of_region_sups);
            }
        }
    }
    return b_hat;
}

inline CommonIntermediateValues compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points,
    const HistogramBase & base,
    const HomogeneousKernels<IntervalKernel> & kernels) {
    const auto nb_regions = points.nb_regions();
    std::vector<Matrix_M_MK1> b_by_region;
    std::vector<MatrixG> g_by_region;
    b_by_region.reserve(nb_regions);
    g_by_region.reserve(nb_regions);
    for(RegionId r = 0; r < nb_regions; ++r) {
        b_by_region.emplace_back(compute_b(points.data_for_region(r), base, kernels.kernels));
        g_by_region.emplace_back(compute_g(points.data_for_region(r), base, kernels.kernels));
    }
    return {std::move(b_by_region), std::move(g_by_region), compute_b_hat(points, base, kernels.kernels)};
}

/******************************************************************************
 * Experimental: heterogeneous kernels (kernel width for each point).
 * Data set is composed of points, kernels.
 */
inline Matrix_M_MK1 compute_b(
    span<const SortedVec<Point>> points,
    const HistogramBase & base,
    span<const std::vector<IntervalKernel>> kernels,
    const std::vector<IntervalKernel> & maximum_width_kernels) {
    assert(points.size() == kernels.size());
    assert(points.size() == maximum_width_kernels.size());
    const auto nb_processes = points.size();
    const auto base_size = base.base_size;
    Matrix_M_MK1 b(nb_processes, base_size);

    const auto sum_sqrt_kernel_width = [](const std::vector<IntervalKernel> & kernels) {
        double sum = 0.;
        for(const auto & kernel : kernels) {
            sum += std::sqrt(kernel.width);
        }
        return sum;
    };
    const auto b_mlk = [&base](
                           const SortedVec<Point> & m_points,
                           const std::vector<IntervalKernel> & m_kernels,
                           IntervalKernel m_maximum_width_kernel,
                           const SortedVec<Point> & l_points,
                           const std::vector<IntervalKernel> & l_kernels,
                           IntervalKernel l_maximum_width_kernel,
                           FunctionBaseId k) {
        assert(m_points.size() == m_kernels.size());
        assert(l_points.size() == l_kernels.size());
        const auto phi_shape = to_shape(base.histogram(k));

        const auto maximum_width_shape =
            convolution(phi_shape, convolution(to_shape(m_maximum_width_kernel), to_shape(l_maximum_width_kernel)));
        const auto union_non_zero_domain = maximum_width_shape.non_zero_domain();

        const auto shape_generator = [&](size_t i_m, size_t i_l) {
            assert(i_m < m_kernels.size());
            assert(i_l < l_kernels.size());
            return convolution(phi_shape, convolution(to_shape(m_kernels[i_m]), to_shape(l_kernels[i_l])));
        };
        return shape::sum_of_point_differences(m_points, l_points, shape_generator, union_non_zero_domain);
    };

    for(ProcessId m = 0; m < nb_processes; ++m) {
        // b0
        b.set_0(m, sum_sqrt_kernel_width(kernels[m]));
        // b_lk
        for(ProcessId l = 0; l < nb_processes; ++l) {
            for(FunctionBaseId k = 0; k < base_size; ++k) {
                const auto v = b_mlk(
                    points[m],
                    kernels[m],
                    maximum_width_kernels[m],
                    points[l],
                    kernels[l],
                    maximum_width_kernels[l],
                    k);
                b.set_lk(m, l, k, v);
            }
        }
    }
    return b;
}

inline MatrixG compute_g(
    span<const SortedVec<Point>> points,
    const HistogramBase & base,
    span<const std::vector<IntervalKernel>> kernels,
    const std::vector<IntervalKernel> & maximum_width_kernels) {
    assert(points.size() == kernels.size());
    assert(points.size() == maximum_width_kernels.size());
    const auto nb_processes = points.size();
    const auto base_size = base.base_size;
    const auto sqrt_delta = std::sqrt(base.delta);
    MatrixG g(nb_processes, base_size);

    g.set_tmax(tmax(points));

    const auto sum_sqrt_kernel_width = [](const std::vector<IntervalKernel> & kernels) {
        double sum = 0.;
        for(const auto & kernel : kernels) {
            sum += std::sqrt(kernel.width);
        }
        return sum;
    };
    for(ProcessId l = 0; l < nb_processes; ++l) {
        /* g_lk = sum_{x_m} integral convolution(w_l,phi_k) (x - x_m) dx.
         * g_lk = sum_{x_m} (integral w_l) (integral phi_k) = sum_{x_m} eta_l sqrt(delta) = |N_m| eta_l sqrt(delta).
         */
        const auto g_lk = sum_sqrt_kernel_width(kernels[l]) * sqrt_delta;
        for(FunctionBaseId k = 0; k < base_size; ++k) {
            g.set_g(l, k, g_lk);
        }
    }

    const auto G_ll2kk2 = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
        assert(points[l].size() == kernels[l].size());
        assert(points[l2].size() == kernels[l2].size());
        const auto & l_points = points[l];
        const auto & l2_points = points[l2];
        const auto & l_kernels = kernels[l];
        const auto & l2_kernels = kernels[l2];

        const auto phi_shape = to_shape(base.histogram(k));
        const auto phi_shape_2 = to_shape(base.histogram(k2));

        const auto maximum_width_shape = cross_correlation(
            convolution(to_shape(maximum_width_kernels[l]), phi_shape),
            convolution(to_shape(maximum_width_kernels[l2]), phi_shape_2));
        const auto union_non_zero_domain = maximum_width_shape.non_zero_domain();

        const auto shape_generator = [&](size_t i_l, size_t i_l2) {
            assert(i_l < l_kernels.size());
            assert(i_l2 < l2_kernels.size());
            return cross_correlation(
                convolution(to_shape(l_kernels[i_l]), phi_shape), convolution(to_shape(l2_kernels[i_l2]), phi_shape_2));
        };
        return shape::sum_of_point_differences(l_points, l2_points, shape_generator, union_non_zero_domain);
    };
    /* G symmetric, only compute for (l2,k2) >= (l,k) (lexicographically).
     *
     * G_{l,l2,k,k2} = integral_x sum_{x_l,x_l2} conv(w_l,phi_k) (x - x_l) conv(w_l2,phi_k2) (x - x_l2) dx.
     * By change of variable x -> x + c*delta: G_{l,l2,k,k2} = G_{l,l2,k+c,k2+c}.
     * Thus for each block of G for (l,l2), we only need to compute 2K+1 values of G.
     * We compute the values on two borders, and copy the values for all inner cells.
     */
    for(ProcessId l = 0; l < nb_processes; ++l) {
        // Case l2 == l: compute for k2 >= k:
        for(FunctionBaseId k = 0; k < base_size; ++k) {
            // Compute G_{l,l,0,k} and copy to G_{l,l,c,k+c} for k in [0,K[.
            const auto v = G_ll2kk2(l, l, FunctionBaseId{0}, k);
            for(size_t c = 0; k + c < base_size; ++c) {
                g.set_G(l, l, FunctionBaseId{c}, FunctionBaseId{k + c}, v);
            }
        }
        // Case l2 > l: compute for all (k,k2):
        for(ProcessId l2 = l; l2 < nb_processes; ++l2) {
            // Compute G_{l,l2,0,0} and copy to G_{l,l2,c,c}.
            {
                const auto v = G_ll2kk2(l, l2, FunctionBaseId{0}, FunctionBaseId{0});
                for(size_t c = 0; c < base_size; ++c) {
                    g.set_G(l, l2, FunctionBaseId{c}, FunctionBaseId{c}, v);
                }
            }
            /* for k in [1,K[:
             * Compute G_{l,l2,0,k} and copy to G_{l,l2,c,k+c}.
             * Compute G_{l,l2,k,0} and copy to G_{l,l2,k+c,c}.
             */
            for(FunctionBaseId k = 1; k < base_size; ++k) {
                const auto v_0k = G_ll2kk2(l, l2, FunctionBaseId{0}, k);
                const auto v_k0 = G_ll2kk2(l, l2, k, FunctionBaseId{0});
                for(size_t c = 0; k + c < base_size; ++c) {
                    g.set_G(l, l2, FunctionBaseId{c}, FunctionBaseId{k + c}, v_0k);
                    g.set_G(l, l2, FunctionBaseId{k + c}, FunctionBaseId{c}, v_k0);
                }
            }
        }
    }
    return g;
}

inline CommonIntermediateValues compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points,
    const HistogramBase & base,
    const HeterogeneousKernels<IntervalKernel> & kernels) {
    const auto nb_regions = points.nb_regions();
    std::vector<Matrix_M_MK1> b_by_region;
    std::vector<MatrixG> g_by_region;
    b_by_region.reserve(nb_regions);
    g_by_region.reserve(nb_regions);
    for(RegionId r = 0; r < nb_regions; ++r) {
        b_by_region.emplace_back(compute_b(
            points.data_for_region(r), base, kernels.kernels.data_for_region(r), kernels.maximum_width_kernels));
        g_by_region.emplace_back(compute_g(
            points.data_for_region(r), base, kernels.kernels.data_for_region(r), kernels.maximum_width_kernels));
    }
    // B_hat too complex to compute
    Matrix_M_MK1 b_hat(points.nb_processes(), base.base_size);
    b_hat.inner.setZero();
    return {std::move(b_by_region), std::move(g_by_region), std::move(b_hat)};
}

/******************************************************************************
 * Haar base, no kernel, not optimized.
 * TODO improve
 */
inline Matrix_M_MK1 compute_b(span<const SortedVec<Point>> points, const HaarBase & base) {
    const size_t nb_processes = points.size();
    const size_t base_size = base.base_size();
    Matrix_M_MK1 b(nb_processes, base_size);

    const auto b_mlk =
        [](const SortedVec<Point> & m_points, const SortedVec<Point> & l_points, const HaarBase::Wavelet & wavelet) {
            // Using linear properties of sum_of_point_differences
            // TODO can be optimized as down_part(scale, pos) = -up_part(scale, pos + 1)
            return sum_of_point_differences(m_points, l_points, up_shape(wavelet)) -
                   sum_of_point_differences(m_points, l_points, down_shape(wavelet));
        };
    for(ProcessId m = 0; m < nb_processes; ++m) {
        // b_0
        b.set_0(m, double(points[m].size()));
        // b_lk
        for(ProcessId l = 0; l < nb_processes; ++l) {
            for(FunctionBaseId k = 0; k < base_size; ++k) {
                const auto v = b_mlk(points[m], points[l], base.wavelet(k));
                b.set_lk(m, l, k, v);
            }
        }
    }
    return b;
}

inline MatrixG compute_g(span<const SortedVec<Point>> points, const HaarBase & base) {
    const size_t nb_processes = points.size();
    const size_t base_size = base.base_size();
    MatrixG g(nb_processes, base_size);

    g.set_tmax(tmax(points));

    for(ProcessId l = 0; l < nb_processes; ++l) {
        for(FunctionBaseId k = 0; k < base_size; ++k) {
            g.set_g(l, k, 0.); // integral_R phi_k = 0
        }
    }

    auto G_ll2kk2 = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
        const auto wavelet1 = base.wavelet(k);
        const auto wavelet2 = base.wavelet(k2);
        // Decompose shape by cross_correlation
        const auto shape_uu = cross_correlation(up_shape(wavelet1), up_shape(wavelet2));
        const auto shape_ud = cross_correlation(up_shape(wavelet1), down_shape(wavelet2));
        const auto shape_du = cross_correlation(down_shape(wavelet1), up_shape(wavelet2));
        const auto shape_dd = cross_correlation(down_shape(wavelet1), down_shape(wavelet2));
        return sum_of_point_differences(points[l], points[l2], shape_uu) +
               sum_of_point_differences(points[l], points[l2], shape_ud) +
               sum_of_point_differences(points[l], points[l2], shape_du) +
               sum_of_point_differences(points[l], points[l2], shape_dd);
    };
    // G symmetric, only compute for (l2,k2) >= (l,k) (lexicographically).
    for(ProcessId l = 0; l < nb_processes; ++l) {
        // Case l2 == l: compute for k2 >= k:
        for(FunctionBaseId k = 0; k < base_size; ++k) {
            for(FunctionBaseId k2 = k; k2 < base_size; ++k2) {
                g.set_G(l, l, k, k2, G_ll2kk2(l, l, k, k2));
            }
        }
        // Case l2 > l: compute for all (k,k2):
        for(ProcessId l2 = l; l2 < nb_processes; ++l2) {
            for(FunctionBaseId k = 0; k < base_size; ++k) {
                for(FunctionBaseId k2 = 0; k2 < base_size; ++k2) {
                    g.set_G(l, l2, k, k2, G_ll2kk2(l, l2, k, k2));
                }
            }
        }
    }
    return g;
}

inline CommonIntermediateValues compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points, const HaarBase & base) {
    const auto nb_regions = points.nb_regions();
    std::vector<Matrix_M_MK1> b_by_region;
    std::vector<MatrixG> g_by_region;
    b_by_region.reserve(nb_regions);
    g_by_region.reserve(nb_regions);
    for(RegionId r = 0; r < nb_regions; ++r) {
        b_by_region.emplace_back(compute_b(points.data_for_region(r), base));
        g_by_region.emplace_back(compute_g(points.data_for_region(r), base));
    }

    // For now ignore the sup part
    Matrix_M_MK1 b_hat(points.nb_processes(), base.base_size());
    b_hat.inner.setZero();
    return {std::move(b_by_region), std::move(g_by_region), std::move(b_hat)};
}

/******************************************************************************
 * Computations common to all cases.
 * TODO naming and doc
 */
struct LassoParameters {
    Matrix_M_MK1 sum_of_b;
    MatrixG sum_of_g;
    Matrix_M_MK1 d;
};

inline LassoParameters compute_lasso_parameters(const CommonIntermediateValues & values, double gamma) {
    const auto check_matrix = [](const Eigen::MatrixXd & m, const string_view what, RegionId r) {
        if(!m.allFinite()) {
#ifndef NDEBUG
            // Print matrix in debug mode
            fmt::print(stderr, "##### Bad values for {}[region={:2}] #####\n", what, r);
            fmt::print(stderr, "{}\n", m);
            fmt::print(stderr, "#########################################\n");
#endif
            throw std::runtime_error(fmt::format("{}[region={}] has non finite values", what, r));
        }
    };

    assert(values.b_by_region.size() == values.g_by_region.size());
    const auto nb_regions = values.b_by_region.size();
    const auto nb_processes = values.b_hat.nb_processes;
    const auto base_size = values.b_hat.base_size;

    /* Generate values combining all regions:
     * - sums of B and G.
     * - V_hat = 1/R^2 * sum_r B[r]^2 (component-wise)
     */
    Matrix_M_MK1 sum_of_b(nb_processes, base_size);
    MatrixG sum_of_g(nb_processes, base_size);
    Matrix_M_MK1 v_hat_r2(nb_processes, base_size); // V_hat * R^2

    sum_of_b.inner.setZero();
    sum_of_g.inner.setZero();
    v_hat_r2.m_lk_values().setZero();

    for(RegionId r = 0; r < nb_regions; ++r) {
        const auto & b = values.b_by_region[r];
        const auto & g = values.g_by_region[r];
        check_matrix(b.inner, "B", r);
        check_matrix(g.inner, "G", r);
        sum_of_b.inner += b.inner;
        v_hat_r2.m_lk_values().array() += b.m_lk_values().array().square();
        sum_of_g.inner += g.inner;
    }

    /* Compute D, the penalty for the lassoshooting.
     * V_hat_mkl is computed without dividing by R^2 so add the division to factor.
     */
    const auto log_factor = std::log(nb_processes + nb_processes * nb_processes * base_size);
    const auto v_hat_factor = (1. / double(nb_regions * nb_regions)) * 2. * gamma * log_factor;
    const auto b_hat_factor = gamma * log_factor / 3.;

#ifndef NDEBUG
    // Compare v_hat and b_hat parts of d.
    {
        auto v_hat_part = (v_hat_factor * v_hat_r2.m_lk_values().array()).sqrt();
        auto b_hat_part = b_hat_factor * values.b_hat.m_lk_values().array();
        const Eigen::IOFormat format(3, 0, "\t"); // 3 = precision in digits, this is enough
        fmt::print(stderr, "##### d = v_hat_part + b_hat_part: value of v_hat_part / b_hat_part #####\n");
        fmt::print(stderr, "{}\n", (v_hat_part / b_hat_part).format(format));
    }
#endif

    Matrix_M_MK1 d(nb_processes, base_size);
    d.m_0_values().setZero(); // No penalty for constant component of estimators
    d.m_lk_values() =
        (v_hat_factor * v_hat_r2.m_lk_values().array()).sqrt() + b_hat_factor * values.b_hat.m_lk_values().array();

    return {std::move(sum_of_b), std::move(sum_of_g), std::move(d)};
}

inline Matrix_M_MK1 compute_estimated_a_with_lasso(const LassoParameters & p, double lambda) {
    assert(p.sum_of_b.inner.allFinite());
    assert(p.sum_of_g.inner.allFinite());
    assert(p.d.inner.allFinite());
    const auto nb_processes = p.sum_of_b.nb_processes;
    const auto base_size = p.sum_of_b.base_size;
    Matrix_M_MK1 a(nb_processes, base_size);
    for(ProcessId m = 0; m < nb_processes; ++m) {
        // lambda is a global multiplier for penalty weights.
        a.values_for_m(m) = lassoshooting(p.sum_of_g.inner, p.sum_of_b.values_for_m(m), p.d.values_for_m(m), lambda);
    }
    return a;
}

inline Matrix_M_MK1 compute_reestimated_a(const LassoParameters & p, const Matrix_M_MK1 & estimated_a) {
    const auto compute_non_zero_index_table = [](const auto & eigen_vec) {
        const Eigen::Index nb_non_zero = eigen_vec.count();
        Eigen::VectorXi table(nb_non_zero);
        Eigen::Index non_zero_seen = 0;
        for(Eigen::Index i = 0; i < eigen_vec.size(); ++i) {
            if(eigen_vec[i] != 0) {
                assert(non_zero_seen < nb_non_zero);
                table[non_zero_seen] = i;
                non_zero_seen += 1;
            }
        }
        assert(non_zero_seen == nb_non_zero);
        return table;
    };

    const auto nb_processes = p.sum_of_b.nb_processes;
    const auto base_size = p.sum_of_b.base_size;
    Matrix_M_MK1 a(nb_processes, base_size);
    a.inner.setZero();
    for(ProcessId m = 0; m < nb_processes; ++m) {
        const auto non_zero_indexes = compute_non_zero_index_table(estimated_a.values_for_m(m));
        const auto nb_non_zero = non_zero_indexes.size();
        if(nb_non_zero > 0) {
            // Build B and G without rows/cols for zeros in a_m.
            const auto b_m = p.sum_of_b.values_for_m(m);
            Eigen::VectorXd restricted_b_m(nb_non_zero);
            for(Eigen::Index i = 0; i < nb_non_zero; ++i) {
                restricted_b_m[i] = b_m[non_zero_indexes[i]];
            }
            Eigen::MatrixXd restricted_g(nb_non_zero, nb_non_zero);
            for(Eigen::Index i = 0; i < nb_non_zero; ++i) {
                for(Eigen::Index j = 0; j < nb_non_zero; ++j) {
                    restricted_g(i, j) = p.sum_of_g.inner(non_zero_indexes[i], non_zero_indexes[j]);
                }
            }
            // Solve linear system without components in 0. Using Cholesky as G is semi-definite positive.
            Eigen::VectorXd restricted_a_m = restricted_g.llt().solve(restricted_b_m);
            if((restricted_g * restricted_a_m).isApprox(restricted_b_m)) {
                // Solution is valid, store it
                auto a_m = a.values_for_m(m);
                for(Eigen::Index i = 0; i < nb_non_zero; ++i) {
                    a_m[non_zero_indexes[i]] = restricted_a_m[i];
                }
            } else {
                throw std::runtime_error(
                    fmt::format("Re-estimation: unable to solve restricted linear system (process {})", m));
            }
        }
    }
    return a;
}
