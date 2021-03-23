#pragma once

#include <Eigen/Cholesky>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "lassoshooting.h"
#include "shape.h"
#include "types.h"

/******************************************************************************
 * Compute B, G, V_hat, B_hat intermediate values.
 *
 * This file defines overloads of compute_intermediate_values for many cases.
 * Each case is defined by the type of arguments:
 * - function base,
 * - kernel configuration.
 * All cases returns the same type of values, but use the specific computation code for this case.
 * The right implementation is then selected depending on the actual types of base and kernel configuration.
 */

// Common struct storing computation results.
struct IntermediateValues {
    Matrix_M_MK1 b;
    MatrixG g;
    Matrix_M_MK1 v_hat;
    Matrix_M_MK1 b_hat;
};

// List of defined overloads for implemented cases
inline std::vector<IntermediateValues> compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points, const HistogramBase & base);
inline std::vector<IntermediateValues> compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points,
    const HistogramBase & base,
    const HomogeneousKernels<IntervalKernel> & kernels);
inline std::vector<IntermediateValues> compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points,
    const HistogramBase & base,
    const HeterogeneousKernels<IntervalKernel> & kernels);
inline std::vector<IntermediateValues> compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points, const HaarBase & base);
inline std::vector<IntermediateValues> compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points,
    const HaarBase & base,
    const HomogeneousKernels<IntervalKernel> & kernels);

inline void print_supported_computation_cases() {
    fmt::print("histogram base with no kernel\n");
    fmt::print("histogram base with homogenous interval kernels\n");
    fmt::print("histogram base with heterogeneous interval kernels\n");
    fmt::print("haar wavelet base with no kernel\n");
    fmt::print("haar wavelet base with homogeneous interval kernels\n");
}

// Overload on base classes : perform selection.
// This emulates pattern matching on sum types.
inline std::vector<IntermediateValues> compute_intermediate_values(
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
        } else if(auto * kernels = dynamic_cast<const HomogeneousKernels<IntervalKernel> *>(&kernel_config)) {
            return compute_intermediate_values(points, *haar, *kernels);
        }
    }

    throw std::runtime_error(
        fmt::format("The combination of '{}' with '{}' is not supported", base.name(), kernel_config.name()));
}

/******************************************************************************
 * Helpers.
 */

/* Conversion of objects to shapes.
 */

inline auto to_shape(IntervalKernel kernel) {
    return shape::scaled(
        normalization_factor(kernel),
        shape::Indicator<Bound::Closed, Bound::Closed>{
            {0., kernel.width},
        });
}
inline auto to_shape(const HistogramBase::Histogram & histogram) {
    return shape::scaled(
        histogram.normalization_factor, shape::Indicator<Bound::Open, Bound::Closed>{histogram.interval});
}

inline shape::Add<std::vector<shape::Polynom<Bound::Open, Bound::Closed>>> to_shape(const HaarBase::Wavelet & wavelet) {
    return {{
        {wavelet.up_part, {wavelet.normalization_factor}},    // Up indicator
        {wavelet.down_part, {-wavelet.normalization_factor}}, // Down indicator
    }};
}

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

// Set G(l,l2,k,k2) = f(l,l2,k,k2). G is symmetric so only compute for (l,k) <= (l2,k2).
template <typename F> void set_G_values(MatrixG & g, size_t nb_processes, size_t base_size, F compute_l_l2_k_k2) {
    for(ProcessId l = 0; l < nb_processes; ++l) {
        // Case l2 == l: compute for k2 >= k:
        for(FunctionBaseId k = 0; k < base_size; ++k) {
            for(FunctionBaseId k2 = k; k2 < base_size; ++k2) {
                g.set_G(l, l, k, k2, compute_l_l2_k_k2(l, l, k, k2));
            }
        }
        // Case l2 > l: compute for all (k,k2):
        for(ProcessId l2 = l; l2 < nb_processes; ++l2) {
            for(FunctionBaseId k = 0; k < base_size; ++k) {
                for(FunctionBaseId k2 = 0; k2 < base_size; ++k2) {
                    g.set_G(l, l2, k, k2, compute_l_l2_k_k2(l, l2, k, k2));
                }
            }
        }
    }
}

/* Set G(l,l2,k,k2) = f(l,l2,k,k2). G is symmetric so only compute for (l,k) <= (l2,k2).
 *
 * G_{l,l2,k,k2} = integral_x sum_{x_l,x_l2} phi_k (x - x_l) phi_k2 (x - x_l2) dx.
 * By change of variable x -> x + c*delta: G_{l,l2,k,k2} = G_{l,l2,k+c,k2+c}.
 * Thus for each block of G for (l,l2), we only need to compute 2K+1 values instead of K^2.
 * We compute the values on two borders, and copy the values for all inner cells.
 */
template <typename F>
void set_G_values_histogram(MatrixG & g, size_t nb_processes, size_t base_size, F compute_l_l2_k_k2) {
    for(ProcessId l = 0; l < nb_processes; ++l) {
        // Case l2 == l: compute for k2 >= k:
        for(FunctionBaseId k = 0; k < base_size; ++k) {
            // Compute G_{l,l,0,k} and copy to G_{l,l,c,k+c} for k in [0,K[.
            const auto v = compute_l_l2_k_k2(l, l, FunctionBaseId{0}, k);
            for(size_t c = 0; k + c < base_size; ++c) {
                g.set_G(l, l, FunctionBaseId{c}, FunctionBaseId{k + c}, v);
            }
        }
        // Case l2 > l: compute for all (k,k2):
        for(ProcessId l2 = l; l2 < nb_processes; ++l2) {
            // Compute G_{l,l2,0,0} and copy to G_{l,l2,c,c}.
            {
                const auto v = compute_l_l2_k_k2(l, l2, FunctionBaseId{0}, FunctionBaseId{0});
                for(size_t c = 0; c < base_size; ++c) {
                    g.set_G(l, l2, FunctionBaseId{c}, FunctionBaseId{c}, v);
                }
            }
            /* for k in [1,K[:
             * Compute G_{l,l2,0,k} and copy to G_{l,l2,c,k+c}.
             * Compute G_{l,l2,k,0} and copy to G_{l,l2,k+c,c}.
             */
            for(FunctionBaseId k = 1; k < base_size; ++k) {
                const auto v_0k = compute_l_l2_k_k2(l, l2, FunctionBaseId{0}, k);
                const auto v_k0 = compute_l_l2_k_k2(l, l2, k, FunctionBaseId{0});
                for(size_t c = 0; k + c < base_size; ++c) {
                    g.set_G(l, l2, FunctionBaseId{c}, FunctionBaseId{k + c}, v_0k);
                    g.set_G(l, l2, FunctionBaseId{k + c}, FunctionBaseId{c}, v_k0);
                }
            }
        }
    }
}

inline double sum_sqrt_kernel_width(const std::vector<IntervalKernel> & kernels) {
    double sum = 0.;
    for(const auto & kernel : kernels) {
        sum += std::sqrt(kernel.width);
    }
    return sum;
}

/******************************************************************************
 * Histogram cases.
 */

inline std::vector<IntermediateValues> compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points, const HistogramBase & base) {
    // Constants
    const size_t base_size = base.base_size;
    const size_t nb_processes = points.nb_processes();
    const size_t nb_regions = points.nb_regions();
    // Compute region values
    auto compute_region_values = [&base, base_size, nb_processes](span<const SortedVec<Point>> points) {
        // B, V_hat
        Matrix_M_MK1 b(nb_processes, base_size);
        Matrix_M_MK1 v_hat(nb_processes, base_size);
        for(ProcessId m = 0; m < nb_processes; ++m) {
            // spontaneous
            b.set_0(m, double(points[m].size()));
            v_hat.set_0(m, double(points[m].size()));
            // lk
            for(ProcessId l = 0; l < nb_processes; ++l) {
                for(FunctionBaseId k = 0; k < base_size; ++k) {
                    const auto phi_k = to_shape(base.histogram(k));
                    auto sums = sum_shape_point_differences(points[m], points[l], phi_k);
                    b.set_lk(m, l, k, sums.non_squared);
                    v_hat.set_lk(m, l, k, sums.squared);
                }
            }
        }
        // B_hat
        // The sup is not affected by translation, so sup(phi_0) = sup(phi_k)
        // Only compute for phi_0 and replicate for all phi_k.
        Matrix_M_MK1 b_hat(nb_processes, base_size);
        const auto phi_0 = to_shape(base.histogram(FunctionBaseId{0}));
        b_hat.m_0_values().setOnes();
        for(ProcessId l = 0; l < nb_processes; ++l) {
            const double b_hat_l = sup_sum_shape_differences_to_points(points[l], phi_0);
            for(ProcessId m = 0; m < nb_processes; ++m) {
                for(FunctionBaseId k = 0; k < base_size; ++k) {
                    b_hat.set_lk(m, l, k, b_hat_l);
                }
            }
        }
        // G
        MatrixG g(nb_processes, base_size);
        const auto sqrt_delta = std::sqrt(base.delta);
        g.set_tmax(tmax(points));
        for(ProcessId l = 0; l < nb_processes; ++l) {
            const auto g_lk = double(points[l].size()) * sqrt_delta;
            for(FunctionBaseId k = 0; k < base_size; ++k) {
                g.set_g(l, k, g_lk);
            }
        }
        auto G_ll2kk2 = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
            const auto shape = cross_correlation(to_shape(base.histogram(k)), to_shape(base.histogram(k2)));
            return sum_shape_point_differences(points[l], points[l2], shape).non_squared;
        };
        set_G_values_histogram(g, nb_processes, base_size, G_ll2kk2);
        // Pack values
        return IntermediateValues{
            std::move(b),
            std::move(g),
            std::move(v_hat),
            std::move(b_hat),
        };
    };
    // All values for all regions
    std::vector<IntermediateValues> regions;
    regions.reserve(nb_regions);
    for(RegionId r = 0; r < nb_regions; ++r) {
        regions.emplace_back(compute_region_values(points.data_for_region(r)));
    }
    return regions;
}

inline std::vector<IntermediateValues> compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points,
    const HistogramBase & base,
    const HomogeneousKernels<IntervalKernel> & kernels) {
    // Constants
    const size_t nb_processes = points.nb_processes();
    const size_t nb_regions = points.nb_regions();
    const size_t base_size = base.base_size;
    // Compute region values
    auto compute_region_values = [&base, &kernels, base_size, nb_processes](span<const SortedVec<Point>> points) {
        // B, V_hat, B_hat
        Matrix_M_MK1 b(nb_processes, base_size);
        Matrix_M_MK1 v_hat(nb_processes, base_size);
        Matrix_M_MK1 b_hat(nb_processes, base_size);
        for(ProcessId m = 0; m < nb_processes; ++m) {
            // spontaneous
            b.set_0(m, double(points[m].size()));
            v_hat.set_0(m, double(points[m].size()));
            b_hat.set_0(m, 1.);
            // lk
            for(ProcessId l = 0; l < nb_processes; ++l) {
                for(FunctionBaseId k = 0; k < base_size; ++k) {
                    const auto shape = convolution(
                        to_shape(kernels.kernels[m]),
                        positive_support(convolution(to_shape(kernels.kernels[l]), to_shape(base.histogram(k)))));
                    auto sums = sum_shape_point_differences(points[m], points[l], shape);
                    b.set_lk(m, l, k, sums.non_squared);
                    v_hat.set_lk(m, l, k, sums.squared);
                    const auto approximated = indicator_approximation(shape);
                    b_hat.set_lk(m, l, k, sup_sum_shape_differences_to_points(points[l], approximated));
                }
            }
        }
        // G
        MatrixG g(nb_processes, base_size);
        g.set_tmax(tmax(points));
        const auto sqrt_delta = std::sqrt(double(base.delta));
        for(ProcessId l = 0; l < nb_processes; ++l) {
            // g_lk = |N_l| \int phi_k \int W_l
            const auto g_lk = double(points[l].size()) * sqrt_delta;
            for(FunctionBaseId k = 0; k < base_size; ++k) {
                g.set_g(l, k, g_lk);
            }
        }
        const auto G_ll2kk2 = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
            const auto shape = cross_correlation(
                positive_support(convolution(to_shape(kernels.kernels[l]), to_shape(base.histogram(k)))),
                positive_support(convolution(to_shape(kernels.kernels[l2]), to_shape(base.histogram(k2)))));
            return sum_shape_point_differences(points[l], points[l2], shape).non_squared;
        };
        set_G_values_histogram(g, nb_processes, base_size, G_ll2kk2);
        // Pack values
        return IntermediateValues{
            std::move(b),
            std::move(g),
            std::move(v_hat),
            std::move(b_hat),
        };
    };
    // Values for all regions
    std::vector<IntermediateValues> regions;
    regions.reserve(nb_regions);
    for(RegionId r = 0; r < nb_regions; ++r) {
        regions.emplace_back(compute_region_values(points.data_for_region(r)));
    }
    return regions;
}

inline std::vector<IntermediateValues> compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points,
    const HistogramBase & base,
    const HeterogeneousKernels<IntervalKernel> & kernels) {
    // Constants
    const size_t nb_processes = points.nb_processes();
    const size_t base_size = base.base_size;
    const size_t nb_regions = points.nb_regions();
    const std::vector<IntervalKernel> & maximum_width_kernels = kernels.maximum_width_kernels;
    // Compute region values
    auto compute_region_values = [&base, &maximum_width_kernels, base_size, nb_processes](
                                     span<const SortedVec<Point>> points,
                                     span<const std::vector<IntervalKernel>> kernels) {
        // B, V_hat, B_hat
        Matrix_M_MK1 b(nb_processes, base_size);
        Matrix_M_MK1 v_hat(nb_processes, base_size);
        Matrix_M_MK1 b_hat(nb_processes, base_size);
        for(ProcessId m = 0; m < nb_processes; ++m) {
            // spontaneous
            b.set_0(m, double(points[m].size()));
            v_hat.set_0(m, double(points[m].size()));
            b_hat.set_0(m, 1.);
            // lk
            for(ProcessId l = 0; l < nb_processes; ++l) {
                for(FunctionBaseId k = 0; k < base_size; ++k) {
                    const auto phi_k = to_shape(base.histogram(k));
                    const auto maximum_width_shape = convolution(
                        to_shape(maximum_width_kernels[m]),
                        positive_support(convolution(to_shape(maximum_width_kernels[l]), phi_k)));
                    const auto union_non_zero_domain = maximum_width_shape.non_zero_domain();

                    const auto shape_generator = [&](size_t i_m, size_t i_l) {
                        assert(i_m < kernels[m].size());
                        assert(i_l < kernels[l].size());
                        return convolution(
                            to_shape(kernels[m][i_m]), positive_support(convolution(to_shape(kernels[l][i_l]), phi_k)));
                    };
                    auto sums = shape::sum_shape_generator_point_differences(
                        points[m], points[l], shape_generator, union_non_zero_domain);
                    b.set_lk(m, l, k, sums.non_squared);
                    v_hat.set_lk(m, l, k, sums.squared);
                    b_hat.set_lk(m, l, k, 0.); // No way to compute this sup
                }
            }
        }
        // G
        MatrixG g(nb_processes, base_size);
        g.set_tmax(tmax(points));
        const double sqrt_delta = std::sqrt(base.delta);
        for(ProcessId l = 0; l < nb_processes; ++l) {
            // g_lk = \int phi_k sum_{N_l} \int W_l
            const auto g_lk = double(points[l].size()) * sqrt_delta;
            for(FunctionBaseId k = 0; k < base_size; ++k) {
                g.set_g(l, k, g_lk);
            }
        }
        const auto G_ll2kk2 = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
            assert(points[l].size() == kernels[l].size());
            assert(points[l2].size() == kernels[l2].size());
            const auto phi_k = to_shape(base.histogram(k));
            const auto phi_k2 = to_shape(base.histogram(k2));

            const auto maximum_width_shape = cross_correlation(
                positive_support(convolution(to_shape(maximum_width_kernels[l]), phi_k)),
                positive_support(convolution(to_shape(maximum_width_kernels[l2]), phi_k2)));
            const auto union_non_zero_domain = maximum_width_shape.non_zero_domain();

            const auto shape_generator = [&](size_t i_l, size_t i_l2) {
                assert(i_l < kernels[l].size());
                assert(i_l2 < kernels[l2].size());
                return cross_correlation(
                    positive_support(convolution(to_shape(kernels[l][i_l]), phi_k)),
                    positive_support(convolution(to_shape(kernels[l2][i_l2]), phi_k2)));
            };
            return shape::sum_shape_generator_point_differences(
                       points[l], points[l2], shape_generator, union_non_zero_domain)
                .non_squared;
        };
        set_G_values_histogram(g, nb_processes, base_size, G_ll2kk2);
        // Pack values
        return IntermediateValues{
            std::move(b),
            std::move(g),
            std::move(v_hat),
            std::move(b_hat),
        };
    };
    // All values for all regions
    std::vector<IntermediateValues> regions;
    regions.reserve(nb_regions);
    for(RegionId r = 0; r < nb_regions; ++r) {
        regions.emplace_back(compute_region_values(points.data_for_region(r), kernels.kernels.data_for_region(r)));
    }
    return regions;
}

/******************************************************************************
 * Haar base
 */
inline std::vector<IntermediateValues> compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points, const HaarBase & base) {
    // Constants
    const size_t nb_processes = points.nb_processes();
    const size_t base_size = base.base_size();
    const auto nb_regions = points.nb_regions();
    // Region values
    auto compute_region_values = [&base, base_size, nb_processes](span<const SortedVec<Point>> points) {
        // B, V_hat, B_hat
        Matrix_M_MK1 b(nb_processes, base_size);
        Matrix_M_MK1 v_hat(nb_processes, base_size);
        Matrix_M_MK1 b_hat(nb_processes, base_size);
        for(ProcessId m = 0; m < nb_processes; ++m) {
            // spontaneous
            b.set_0(m, double(points[m].size()));
            v_hat.set_0(m, double(points[m].size()));
            b_hat.set_0(m, 1.);
            // lk
            for(ProcessId l = 0; l < nb_processes; ++l) {
                for(FunctionBaseId k = 0; k < base_size; ++k) {
                    const auto phi_k = to_shape(base.wavelet(k));
                    auto sums = sum_shape_point_differences(points[m], points[l], phi_k);
                    b.set_lk(m, l, k, sums.non_squared);
                    v_hat.set_lk(m, l, k, sums.squared);
                    b_hat.set_lk(m, l, k, 0.); // TODO
                }
            }
        }
        // G
        MatrixG g(nb_processes, base_size);
        g.set_tmax(tmax(points));
        for(ProcessId l = 0; l < nb_processes; ++l) {
            for(FunctionBaseId k = 0; k < base_size; ++k) {
                g.set_g(l, k, 0.); // integral_R phi_k = 0
            }
        }
        auto G_ll2kk2 = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
            const auto shape = cross_correlation(to_shape(base.wavelet(k)), to_shape(base.wavelet(k2)));
            return sum_shape_point_differences(points[l], points[l2], shape).non_squared;
        };
        set_G_values(g, nb_processes, base_size, G_ll2kk2);
        // Pack values
        return IntermediateValues{
            std::move(b),
            std::move(g),
            std::move(v_hat),
            std::move(b_hat),
        };
    };
    // All values for all regions
    std::vector<IntermediateValues> regions;
    regions.reserve(nb_regions);
    for(RegionId r = 0; r < nb_regions; ++r) {
        regions.emplace_back(compute_region_values(points.data_for_region(r)));
    }
    return regions;
}

inline std::vector<IntermediateValues> compute_intermediate_values(
    const DataByProcessRegion<SortedVec<Point>> & points,
    const HaarBase & base,
    const HomogeneousKernels<IntervalKernel> & kernels) {
    // Constants
    const size_t nb_processes = points.nb_processes();
    const size_t nb_regions = points.nb_regions();
    const size_t base_size = base.base_size();
    // Compute region values
    auto compute_region_values = [&base, &kernels, base_size, nb_processes](span<const SortedVec<Point>> points) {
        // B, V_hat, B_hat
        Matrix_M_MK1 b(nb_processes, base_size);
        Matrix_M_MK1 v_hat(nb_processes, base_size);
        Matrix_M_MK1 b_hat(nb_processes, base_size);
        for(ProcessId m = 0; m < nb_processes; ++m) {
            // spontaneous
            b.set_0(m, double(points[m].size()));
            v_hat.set_0(m, double(points[m].size()));
            b_hat.set_0(m, 1.);
            // lk
            for(ProcessId l = 0; l < nb_processes; ++l) {
                for(FunctionBaseId k = 0; k < base_size; ++k) {
                    const auto shape = convolution(
                        to_shape(kernels.kernels[m]),
                        positive_support(convolution(to_shape(kernels.kernels[l]), to_shape(base.wavelet(k)))));
                    auto sums = sum_shape_point_differences(points[m], points[l], shape);
                    b.set_lk(m, l, k, sums.non_squared);
                    v_hat.set_lk(m, l, k, sums.squared);
                    b_hat.set_lk(m, l, k, 0.); // TODO ?
                }
            }
        }
        // G
        MatrixG g(nb_processes, base_size);
        g.set_tmax(tmax(points));
        for(ProcessId l = 0; l < nb_processes; ++l) {
            // g_lk = |N_l| \int phi_k \int W_l
            for(FunctionBaseId k = 0; k < base_size; ++k) {
                g.set_g(l, k, 0.);
            }
        }
        const auto G_ll2kk2 = [&](ProcessId l, ProcessId l2, FunctionBaseId k, FunctionBaseId k2) {
            const auto shape = cross_correlation(
                positive_support(convolution(to_shape(kernels.kernels[l]), to_shape(base.wavelet(k)))),
                positive_support(convolution(to_shape(kernels.kernels[l2]), to_shape(base.wavelet(k2)))));
            return sum_shape_point_differences(points[l], points[l2], shape).non_squared;
        };
        set_G_values(g, nb_processes, base_size, G_ll2kk2);
        // Pack values
        return IntermediateValues{
            std::move(b),
            std::move(g),
            std::move(v_hat),
            std::move(b_hat),
        };
    };
    // Values for all regions
    std::vector<IntermediateValues> regions;
    regions.reserve(nb_regions);
    for(RegionId r = 0; r < nb_regions; ++r) {
        regions.emplace_back(compute_region_values(points.data_for_region(r)));
    }
    return regions;
}

/******************************************************************************
 * Computations common to all cases.
 * TODO naming and doc
 */
struct LassoParameters {
    Matrix_M_MK1 sum_of_b;
    MatrixG sum_of_g;
    Matrix_M_MK1 d;

    // For debugging
    Matrix_M_MK1 sum_of_v_hat; // Without 1/R^2 factor
    Matrix_M_MK1 sum_of_b_hat;
};

inline LassoParameters compute_lasso_parameters(
    const std::vector<IntermediateValues> & values_by_region, double gamma) {
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

    const auto nb_regions = values_by_region.size();
    assert(nb_regions > 0);
    const auto nb_processes = values_by_region[0].b.nb_processes;
    const auto base_size = values_by_region[0].b.base_size;

    /* Sum values from all regions.
     */
    Matrix_M_MK1 sum_of_b(nb_processes, base_size);
    MatrixG sum_of_g(nb_processes, base_size);
    Matrix_M_MK1 sum_of_b_hat(nb_processes, base_size);
    Matrix_M_MK1 sum_of_v_hat(nb_processes, base_size);

    sum_of_b.inner.setZero();
    sum_of_g.inner.setZero();
    sum_of_b_hat.inner.setZero();
    sum_of_v_hat.inner.setZero();

    for(RegionId r = 0; r < nb_regions; ++r) {
        const IntermediateValues & values = values_by_region[r];
        check_matrix(values.b.inner, "B", r);
        check_matrix(values.g.inner, "G", r);
        check_matrix(values.v_hat.inner, "V_hat", r);
        check_matrix(values.b_hat.inner, "B_hat", r);

        sum_of_b.inner += values.b.inner;
        sum_of_g.inner += values.g.inner;
        sum_of_v_hat.inner += values.v_hat.inner;
        sum_of_b_hat.inner += values.b_hat.inner;
    }

    /* Compute D, the penalty for the lassoshooting.
     * sum_of_V_hat is computed without dividing by R^2 so add the division to factor.
     */
    const auto log_factor = std::log(nb_processes + nb_processes * nb_processes * base_size);
    const auto v_hat_factor = (1. / double(nb_regions * nb_regions)) * 2. * gamma * log_factor;
    const auto b_hat_factor = gamma * log_factor / 3.;

    Matrix_M_MK1 d(nb_processes, base_size);
    d.inner.array() = (v_hat_factor * sum_of_v_hat.inner.array()).sqrt() + b_hat_factor * sum_of_b_hat.inner.array();

#ifndef NDEBUG
    // Compare v_hat and b_hat parts of d.
    {
        auto v_hat_part = (v_hat_factor * sum_of_v_hat.inner.array()).sqrt();
        auto b_hat_part = b_hat_factor * sum_of_b_hat.inner.array();
        const Eigen::IOFormat format(3, 0, "\t"); // 3 = precision in digits, this is enough
        fmt::print(stderr, "##### v_hat_part / b_hat_part, for d = v_hat_part + b_hat_part #####\n");
        fmt::print(stderr, "{}\n", (v_hat_part / b_hat_part).format(format));
    }
#endif

    return {std::move(sum_of_b), std::move(sum_of_g), std::move(d), std::move(sum_of_v_hat), std::move(sum_of_b_hat)};
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
