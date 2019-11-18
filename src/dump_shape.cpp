#include "computations.h"
#include "shape.h"

#include <cstdio>

using namespace shape;

namespace fmt {
template <Bound lb, Bound rb> struct formatter<Interval<lb, rb>> : formatter<double> {
    static constexpr char delimiter_left() { return lb == Bound::Open ? ']' : '['; }
    static constexpr char delimiter_right() { return rb == Bound::Open ? '[' : ']'; }

    template <typename FormatContext> auto format(const Interval<lb, rb> & interval, FormatContext & ctx) {
        return format_to(ctx.out(), "{}{}, {}{}", delimiter_left(), interval.left, interval.right, delimiter_right());
    }
};
} // namespace fmt

inline void print_indent(FILE * out, std::size_t n) {
    while(n > 0) {
        std::fputc(' ', out);
        std::fputc(' ', out);
        n -= 1;
    }
}

template <typename Inner> void print_shape_details(FILE * out, const Scaled<Inner> & scaled, std::size_t indent) {
    print_indent(out, indent);
    fmt::print(out, "Scaling {} {{\n", scaled.scale);
    print_shape_details(out, scaled.inner, indent + 1);
    print_indent(out, indent);
    fmt::print(out, "}}\n");
}
template <typename T> void print_shape_details(FILE * out, const Add<std::vector<T>> & add, std::size_t indent) {
    print_indent(out, indent);
    fmt::print(out, "Add<{}> {{\n", add.components.size());
    for(const T & component : add.components) {
        print_shape_details(out, component, indent + 1);
    }
    print_indent(out, indent);
    fmt::print(out, "}}\n");
}
template <Bound lb, Bound rb> void print_shape_details(FILE * out, const Polynom<lb, rb> & p, std::size_t indent) {
    print_indent(out, indent);
    fmt::print(out, "Polynom {} {{{}}}\n", p.interval, fmt::join(p.coefficients, ", "));
}
template <typename Shape> void print_shape_details(FILE * out, const Shape & shape) {
    print_shape_details(out, shape, 0);
}

template <typename Shape> void dump_shape(const Shape & shape) {
    print_shape_details(stderr, shape);
    // Plot shape to stdout
    fmt::print(stdout, "x\tshape\n");
    const auto domain = shape.non_zero_domain();
    const auto domain_size = domain.right - domain.left;
    const auto margins = std::max(domain_size / 10., 10.);
    const auto increment = (domain_size + 2 * margins) / 1000.;
    for(Point x = domain.left - margins; x < domain.right + margins; x += increment) {
        fmt::print(stdout, "{}\t{}\n", x, shape(x));
    }
}

struct Case {
    string_view description;
    void (*action)();
};

int main(int argc, const char * argv[]) {
    const Case cases[] = {
        {
            "convolution(small_indicator, indicator)",
            []() {
                const auto small = Indicator<Bound::Closed, Bound::Closed>{{-0.1, 0.1}};
                const auto indicator = Indicator<Bound::Closed, Bound::Closed>{{-50., 50.}};
                dump_shape(convolution(small, indicator));
            },
        },
        {
            "convolution(indicator, indicator)",
            []() {
                const auto indicator = Indicator<Bound::Closed, Bound::Closed>{{-50., 50.}};
                dump_shape(convolution(indicator, indicator));
            },
        },
        {
            "convolution(IntervalKernel{100}, HistogramBase{5, 1000}.histogram(0))",
            []() {
                const auto kernel = IntervalKernel{100};
                const auto phi = HistogramBase{5, 1000}.histogram(0);
                dump_shape(convolution(to_shape(kernel), to_shape(phi)));
            },
        },
        {
            "convolution(IntervalKernel{100}, HistogramBase{5, 1000}.histogram(0), IntervalKernel{200})",
            []() {
                const auto kernel = IntervalKernel{100};
                const auto kernel2 = IntervalKernel{200};
                const auto phi = HistogramBase{5, 1000}.histogram(0);
                dump_shape(convolution(convolution(to_shape(kernel), to_shape(phi)), to_shape(kernel2)));
            },
        },
        {
            "convolution(IntervalKernel{300}, HistogramBase{5, 1000}.histogram(0), IntervalKernel{300})",
            []() {
                const auto kernel = IntervalKernel{300};
                const auto phi = HistogramBase{5, 1000}.histogram(0);
                dump_shape(convolution(convolution(to_shape(kernel), to_shape(phi)), to_shape(kernel)));
            },
        },
        {
            "cross_correlation(convolution(IntervalKernel{100}, HistogramBase{5, 1000}.histogram(0)), "
            "convolution(IntervalKernel{200}, HistogramBase{5, 1000}.histogram(3)))",
            []() {
                const auto base = HistogramBase{5, 1000};
                const auto kernel = IntervalKernel{100};
                const auto kernel2 = IntervalKernel{200};
                const auto phi = base.histogram(0);
                const auto phi2 = base.histogram(3);
                dump_shape(cross_correlation(
                    convolution(to_shape(kernel), to_shape(phi)), convolution(to_shape(kernel2), to_shape(phi2))));
            },
        },
        {
            "cross_correlation(convolution(IntervalKernel{0.1}, HistogramBase{5, 1000}.histogram(0)), "
            "convolution(IntervalKernel{100}, HistogramBase{5, 1000}.histogram(3)))",
            []() {
                const auto base = HistogramBase{5, 10000};
                const auto kernel = IntervalKernel{0.1};
                const auto kernel2 = IntervalKernel{100};
                const auto phi = base.histogram(0);
                const auto phi2 = base.histogram(3);
                dump_shape(cross_correlation(
                    convolution(to_shape(kernel), to_shape(phi)), convolution(to_shape(kernel2), to_shape(phi2))));
            },
        },
        {
            "Polynom [-10., 10.] {0., -100., 0., 1.}",
            []() {
                auto q = Polynom<Bound::Closed, Bound::Closed>{
                    {-10., 10.},
                    {0., -100., 0., 1.},
                };
                dump_shape(q);
            },
        },
        {
            "convolution(Polynom [-8., 8.] {64., 0., -1.}, Polynom [-10., 10.] {0., -100., 0., 1.})",
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
        },
        {
            "positive_support(convolution(IntervalKernel{100}, HistogramBase{5, 1000}.histogram(0)))",
            []() {
                const auto kernel = IntervalKernel{100};
                const auto phi = HistogramBase{5, 1000}.histogram(0);
                dump_shape(positive_support(convolution(to_shape(kernel), to_shape(phi))));
            },
        },
        {
            "convolution(positive_support(convolution(IntervalKernel{100}, HistogramBase{5, 1000}.histogram(0))), "
            "IntervalKernel{200})",
            []() {
                const auto kernel = IntervalKernel{100};
                const auto kernel2 = IntervalKernel{200};
                const auto phi = HistogramBase{5, 1000}.histogram(0);
                dump_shape(
                    convolution(positive_support(convolution(to_shape(kernel), to_shape(phi))), to_shape(kernel2)));
            },
        },
        {
            "cross_correlation("
            "positive_support(convolution(IntervalKernel{100}, HistogramBase{5, 1000}.histogram(0))), "
            "positive_support(convolution(IntervalKernel{200}, HistogramBase{5, 1000}.histogram(0)))"
            ")",
            []() {
                const auto kernel = IntervalKernel{100};
                const auto kernel2 = IntervalKernel{200};
                const auto phi = HistogramBase{5, 1000}.histogram(0);
                const auto phi2 = HistogramBase{5, 1000}.histogram(3);
                dump_shape(cross_correlation(
                    positive_support(convolution(to_shape(kernel), to_shape(phi))),
                    positive_support(convolution(to_shape(kernel2), to_shape(phi2)))));
            },
        },
        {
            "Haar{2, 1000}.wavelet(1, 0)",
            []() {
                dump_shape(to_shape(HaarBase{2, 1000}.wavelet(1, 0)));
            },
        },
        {
            "convolution(IntervalKernel{100}, Haar{2, 1000}.wavelet(1, 0))",
            []() {
                const auto kernel = IntervalKernel{100};
                const auto phi = HaarBase{2, 1000}.wavelet(1, 0);
                dump_shape(convolution(to_shape(kernel), to_shape(phi)));
            },
        },
        {
            "positive_support(convolution(IntervalKernel{100}, Haar{2, 1000}.wavelet(1, 0)))",
            []() {
                const auto kernel = IntervalKernel{100};
                const auto phi = HaarBase{2, 1000}.wavelet(1, 0);
                dump_shape(positive_support(convolution(to_shape(kernel), to_shape(phi))));
            },
        },
        {
            "convolution("
            "positive_support(convolution(IntervalKernel{100}, Haar{2, 1000}.wavelet(1, 0))),"
            "IntervalKernel{200})",
            []() {
                const auto kernel = IntervalKernel{100};
                const auto kernel2 = IntervalKernel{200};
                const auto phi = HaarBase{2, 1000}.wavelet(1, 0);
                dump_shape(
                    convolution(positive_support(convolution(to_shape(kernel), to_shape(phi))), to_shape(kernel2)));
            },
        },
        {
            "cross_correlation("
            "positive_support(convolution(IntervalKernel{100}, Haar{2, 1000}.wavelet(1, 0))),"
            "positive_support(convolution(IntervalKernel{200}, Haar{2, 1000}.wavelet(1, 1))))",
            []() {
                const auto kernel = IntervalKernel{100};
                const auto kernel2 = IntervalKernel{200};
                const auto phi = HaarBase{2, 1000}.wavelet(1, 0);
                const auto phi2 = HaarBase{2, 1000}.wavelet(1, 1);
                dump_shape(cross_correlation(
                    positive_support(convolution(to_shape(kernel), to_shape(phi))),
                    positive_support(convolution(to_shape(kernel2), to_shape(phi2)))));
            },
        },
    };
    const auto nb_cases = std::distance(std::begin(cases), std::end(cases));

    if(argc == 2) {
        const int i = std::atoi(argv[1]);
        if(0 <= i && i < nb_cases) {
            cases[i].action();
            return 0;
        }
    }
    fmt::print(stderr, "Usage: dump_shape [0-{}]\n\n", nb_cases - 1);
    for(int i = 0; i < nb_cases; ++i) {
        fmt::print(stderr, "[{:02}] {}\n", i, cases[i].description);
    }
    return int(nb_cases);
}
