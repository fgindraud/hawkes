#include "computations.h"
#include "input.h"
#include "shape.h"

template <typename Shape> void dump_shape(const Shape & shape) {
    fmt::print(stdout, "x\tshape\n");
    const auto domain = shape.non_zero_domain();
    const auto domain_size = domain.right - domain.left;
    const auto margins = std::max(domain_size / 10., 10.);
    const auto increment = (domain_size + 2 * margins) / 1000.;
    for(Point x = domain.left - margins; x < domain.right + margins; x += increment) {
        fmt::print(stdout, "{}\t{}\n", x, shape(x));
    }
}

using namespace shape;

int main(int argc, const char * argv[]) {
    using DumpCase = void (*)();

    const DumpCase cases[] = {
        []() {
            const auto empty = Indicator<Bound::Closed, Bound::Closed>{{0., 0.}};
            const auto indicator = Indicator<Bound::Closed, Bound::Closed>{{-50., 50.}};
            dump_shape(convolution(empty, indicator));
        },
        []() {
            const auto small = Indicator<Bound::Closed, Bound::Closed>{{-0.1, 0.1}};
            const auto indicator = Indicator<Bound::Closed, Bound::Closed>{{-50., 50.}};
            dump_shape(convolution(small, indicator));
        },
        []() {
            const auto indicator = Indicator<Bound::Closed, Bound::Closed>{{-50., 50.}};
            dump_shape(convolution(indicator, indicator));
        },
        []() {
            const auto kernel = IntervalKernel{100};
            const auto phi = HistogramBase{5, 1000}.histogram(0);
            dump_shape(convolution(to_shape(kernel), to_shape(phi)));
        },
        []() {
            const auto kernel = IntervalKernel{100};
            const auto kernel2 = IntervalKernel{200};
            const auto phi = HistogramBase{5, 1000}.histogram(0);
            dump_shape(convolution(convolution(to_shape(kernel), to_shape(phi)), to_shape(kernel2)));
        },
        []() {
            const auto kernel = IntervalKernel{300};
            const auto phi = HistogramBase{5, 1000}.histogram(0);
            dump_shape(convolution(convolution(to_shape(kernel), to_shape(phi)), to_shape(kernel)));
        },
        []() {
            const auto base = HistogramBase{5, 1000};
            const auto kernel = IntervalKernel{100};
            const auto kernel2 = IntervalKernel{200};
            const auto phi = base.histogram(0);
            const auto phi2 = base.histogram(3);
            dump_shape(convolution(
                convolution(to_shape(kernel), to_shape(phi)), convolution(to_shape(kernel2), to_shape(phi2))));
        },
        []() {
            const auto base = HistogramBase{5, 10000};
            const auto kernel = IntervalKernel{0.1};
            const auto kernel2 = IntervalKernel{100};
            const auto phi = base.histogram(0);
            const auto phi2 = base.histogram(3);
            dump_shape(convolution(
                convolution(to_shape(kernel), to_shape(phi)), convolution(to_shape(kernel2), to_shape(phi2))));
        },
        []() {
            auto q = Polynom<Bound::Closed, Bound::Closed>{
                {-10., 10.},
                {0., -100., 0., 1.},
            };
            dump_shape(q);
        },
        []() {
            auto p = Polynom<Bound::Closed, Bound::Closed>{
                {-8., 8.},
                {64., 0., -1.},
            };
            auto q = Polynom<Bound::Closed, Bound::Closed>{
                {-10., 10.},
                {0., -100., 0., 1.},
            };
            dump_shape(convolution(p, q));
        },
    };
    const auto nb_cases = std::distance(std::begin(cases), std::end(cases));

    if(argc == 2) {
        const int i = std::atoi(argv[1]);
        if(0 <= i && i < nb_cases) {
            cases[i]();
            return 0;
        }
    }
    return int(nb_cases);
}
