// Copyright 2021 PEI WeiCheng and JIANG YuYan

#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/euler/ausm.hpp"
#include "mini/riemann/euler/hllc.hpp"

namespace mini {
namespace riemann {
namespace euler {

using Gas = IdealGas<double, 1.4>;

template <class Solver>
auto run() {
  using Primitive = typename Solver::Primitive;
  using Flux = typename Solver::Flux;
  Solver solver;
  Flux flux{};
  {
  Primitive left{1.0, 0.0, 1.0}, right{0.125, 0.0, 0.1};
  flux += solver.GetFluxUpwind(left, right);
  flux += solver.GetFluxUpwind(right, left);
  } {
  Primitive left{5.99924, 19.5975, 460.894}, right{5.99242, 6.19633, 46.0950};
  flux += solver.GetFluxUpwind(left, right);
  } {
  Primitive left{1.0, 0.0, 1000}, right{1.0, 0.0, 0.01};
  flux += solver.GetFluxUpwind(left, right);
  } {
  Primitive left{1.0, 0.0, 0.01}, right{1.0, 0.0, 100};
  flux += solver.GetFluxUpwind(left, right);
  } {
  Primitive left{1.0, -2.0, 0.4}, right{1.0, +2.0, 0.4};
  flux += solver.GetFluxUpwind(left, right);
  } {
  Primitive left{1.0, -4.0, 0.4}, right{1.0, +4.0, 0.4};
  flux += solver.GetFluxUpwind(left, right);
  }
  return flux;
}

class TestPerformance : public ::testing::Test {
 public:
#ifdef NDEBUG
  int n = 1'000'000;
#else
  int n = 100'000;
#endif
};
TEST_F(TestPerformance, TestAusm) {
  for (int i = 0; i < n; ++i) {
    run<Ausm<Gas, 1>>();
  }
}
TEST_F(TestPerformance, TestHartenLaxLeerContact) {
  for (int i = 0; i < n; ++i) {
    run<HartenLaxLeerContact<Gas, 1>>();
  }
}
TEST_F(TestPerformance, TestExact) {
  for (int i = 0; i < n; ++i) {
    run<Exact<Gas, 1>>();
  }
}

}  // namespace euler
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
