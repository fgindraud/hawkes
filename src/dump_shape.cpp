#include "computations.h"
#include "input.h"
#include "shape.h"

template <typename Shape> void dump_shape (const Shape & shape) {
	fmt::print (stdout, "x\tshape\n");
	const auto domain = shape.non_zero_domain ();
	const auto margins = std::max ((domain.right - domain.left) / 10, 10);
	for (Point x = domain.left - margins; x < domain.right + margins; ++x) {
		fmt::print (stdout, "{}\t{}\n", x, shape (x));
	}
}

#include <limits>
#include <tuple>
namespace shape {
template <typename... Shapes> struct Add {
	std::tuple<Shapes...> shapes;
	ClosedInterval<Point> nzd;

	ClosedInterval<Point> non_zero_domain () const { return nzd; }
	auto operator() (Point x) const {
		return std::apply ([x](const Shapes &... shapes) { return (shapes (x) + ...); }, shapes);
	}
};
template <typename... Shapes> inline Add<Shapes...> add (const Shapes &... shapes) {
	const Point left = std::min ({(shapes.non_zero_domain ().left)...});
	const Point right = std::max ({(shapes.non_zero_domain ().right)...});
	return {std::make_tuple (shapes...), {left, right}};
}
template <typename Shape> inline auto convolution (const Trapezoid & left, const Shape & right) {
	return add (convolution (component (left, Trapezoid::LeftTriangle{}), right),
	            convolution (component (left, Trapezoid::CentralBlock{}), right),
	            convolution (component (left, Trapezoid::RightTriangle{}), right));
}
inline auto convolution (const Trapezoid & left, const Trapezoid & right) {
	const auto left_part = Trapezoid::LeftTriangle{};
	const auto central_part = Trapezoid::CentralBlock{};
	const auto right_part = Trapezoid::RightTriangle{};
	return add (
	    //
	    convolution (component (left, left_part), component (right, left_part)),
	    convolution (component (left, central_part), component (right, left_part)),
	    convolution (component (left, right_part), component (right, left_part)),
	    //
	    convolution (component (left, left_part), component (right, central_part)),
	    convolution (component (left, central_part), component (right, central_part)),
	    convolution (component (left, right_part), component (right, central_part)),
	    //
	    convolution (component (left, left_part), component (right, right_part)),
	    convolution (component (left, central_part), component (right, right_part)),
	    convolution (component (left, right_part), component (right, right_part)));
}
} // namespace shape

using namespace shape;

int main (int argc, const char * argv[]) {
	using DumpCase = void (*) ();

	const DumpCase cases[] = {
	    []() {
		    const auto empty = IntervalIndicator::with_width (0);
		    const auto interval = IntervalIndicator::with_width (100);
		    dump_shape (convolution (empty, interval));
	    },
	    []() {
		    const auto small = IntervalIndicator::with_width (2);
		    const auto interval = IntervalIndicator::with_width (100);
		    dump_shape (convolution (small, interval));
	    },
	    []() {
		    const auto interval = IntervalIndicator::with_width (100);
		    dump_shape (convolution (interval, interval));
	    },
	    []() {
		    const auto kernel = IntervalKernel (100);
		    const auto phi = HistogramBase (5, 1000).interval (0);
		    dump_shape (convolution (to_shape (kernel), to_shape (phi)));
	    },
	    []() {
		    const auto kernel = IntervalKernel (100);
		    const auto kernel2 = IntervalKernel (200);
		    const auto phi = HistogramBase (5, 1000).interval (0);
		    dump_shape (convolution (convolution (to_shape (kernel), to_shape (phi)), to_shape (kernel2)));
	    },
	    []() {
		    const auto kernel = IntervalKernel (300);
		    const auto phi = HistogramBase (5, 1000).interval (0);
		    dump_shape (convolution (convolution (to_shape (kernel), to_shape (phi)), to_shape (kernel)));
	    },
	    []() {
		    const auto base = HistogramBase (5, 1000);
		    const auto kernel = IntervalKernel (100);
		    const auto kernel2 = IntervalKernel (200);
		    const auto phi = base.interval (0);
		    const auto phi2 = base.interval (3);
		    dump_shape (convolution (convolution (to_shape (kernel), to_shape (phi)),
		                             convolution (to_shape (kernel2), to_shape (phi2))));
	    },
	    []() {
		    const auto base = HistogramBase (5, 10000);
		    const auto kernel = IntervalKernel (2);
		    const auto kernel2 = IntervalKernel (100);
		    const auto phi = base.interval (0);
		    const auto phi2 = base.interval (3);
		    dump_shape (convolution (convolution (to_shape (kernel), to_shape (phi)),
		                             convolution (to_shape (kernel2), to_shape (phi2))));
	    },
	};
	const auto nb_cases = std::distance (std::begin (cases), std::end (cases));

	if (argc == 2) {
		const int i = std::atoi (argv[1]);
		if (0 <= i && i < nb_cases) {
			cases[i]();
			return 0;
		}
	}
	return int(nb_cases);
}
