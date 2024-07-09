//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cstdlib>
#include <cmath>
#include <iomanip>

#include "mini/integrator/function.hpp"
#include "mini/integrator/legendre.hpp"
#include "mini/integrator/lobatto.hpp"
#include "mini/integrator/quadrangle.hpp"
#include "mini/integrator/hexahedron.hpp"
#include "mini/coordinate/quadrangle.hpp"
#include "mini/coordinate/hexahedron.hpp"
#include "mini/basis/linear.hpp"
#include "mini/polynomial/projection.hpp"
#include "mini/polynomial/hexahedron.hpp"
#include "mini/constant/index.hpp"
#include "mini/algebra/eigen.hpp"
#include "mini/rand.hpp"

#include "gtest/gtest.h"

using namespace mini::constant::index;

class TestPolynomialHexahedron : public ::testing::Test {
 protected:
  static constexpr int kTrials = 1 << 5;
  static constexpr int kComponents = 11;
  using Scalar = double;
  using Global = mini::algebra::Vector<Scalar, 3>;
  using Local = Global;
  using Value = mini::algebra::Vector<Scalar, kComponents>;
  using Gradient = mini::algebra::Matrix<Scalar, 3, kComponents>;
  using Hessian = mini::algebra::Matrix<Scalar, 6, kComponents>;

  static Value coeff_;

  void SetUp() override {
    std::srand(31415926);
  }

  static double rand_f(double a = -1., double b = 1.) {
    return mini::rand::uniform(a, b);
  }

  static Value GetExactValue(Global const& xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    Value value{ 0, 1, x, y, z, x*x/2, x * y, x * z, y*y/2, y * z, z*z/2 };
    for (int k = 0; k < kComponents; ++k) {
      value[k] *= coeff_[k];
    }
    return value;
  }

  static Gradient GetExactGradient(Global const& xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    Gradient grad;
    grad << 0, 0, 1, 0, 0, x, y, z, 0, 0, 0,
            0, 0, 0, 1, 0, 0, x, 0, y, z, 0,
            0, 0, 0, 0, 1, 0, 0, x, 0, y, z;
    for (int k = 0; k < kComponents; ++k) {
      grad.col(k) *= coeff_[k];
    }
    return grad;
  }

  static Hessian GetExactHessian(Global const& xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    Hessian hess;
    // Value value{ 0, 1, x, y, z, x*x/2, x * y, x * z, y*y/2, y * z, z*z/2 };
    hess << 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    for (int k = 0; k < kComponents; ++k) {
      hess.col(k) *= coeff_[k];
    }
    return hess;
  };
};

typename TestPolynomialHexahedron::Value
TestPolynomialHexahedron::coeff_;

class TestPolynomialHexahedronProjection : public TestPolynomialHexahedron {
 protected:
  using IntegratorX = mini::integrator::Legendre<Scalar, 4>;
  using Integrator = mini::integrator::Hexahedron<IntegratorX, IntegratorX, IntegratorX>;
  using Coordinate = mini::coordinate::Hexahedron8<Scalar>;
  using Basis = mini::basis::OrthoNormal<Scalar, 3, 2>;
  using Coord = typename Basis::Coord;
  using ScalarProjection = mini::polynomial::Projection<Scalar, 3, 2, 1>;
  using VectorProjection = mini::polynomial::Projection<Scalar, 3, 2, 11>;
};
TEST_F(TestPolynomialHexahedronProjection, OrthoNormal) {
  // build a hexa-integrator
  auto coordinate = Coordinate {
    Coord(-1, -1, -1), Coord(+1, -1, -1), Coord(+1, +1, -1), Coord(-1, +1, -1),
    Coord(-1, -1, +1), Coord(+1, -1, +1), Coord(+1, +1, +1), Coord(-1, +1, +1),
  };
  auto integrator = Integrator(coordinate);
  // build an orthonormal basis on it
  auto basis = Basis(integrator);
  // check orthonormality
  double residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    typename Basis::MatNxN prod = col * col.transpose();
    return prod;
  }, integrator) - Basis::MatNxN::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
  // build another hexa-integrator
  Coord shift = {-1, 2, 3};
  coordinate = Coordinate {
    coordinate.GetGlobal(0) + shift,
    coordinate.GetGlobal(1) + shift,
    coordinate.GetGlobal(2) + shift,
    coordinate.GetGlobal(3) + shift,
    coordinate.GetGlobal(4) + shift,
    coordinate.GetGlobal(5) + shift,
    coordinate.GetGlobal(6) + shift,
    coordinate.GetGlobal(7) + shift,
  };
  integrator = Integrator(coordinate);
  // build another orthonormal basis on it
  basis = Basis(integrator);
  // check orthonormality
  residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    typename Basis::MatNxN prod = col * col.transpose();
    return prod;
  }, integrator) - Basis::MatNxN::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
}
TEST_F(TestPolynomialHexahedronProjection, Projection) {
  auto coordinate = Coordinate{
    Coord(-1, -1, -1), Coord(+1, -1, -1), Coord(+1, +1, -1), Coord(-1, +1, -1),
    Coord(-1, -1, +1), Coord(+1, -1, +1), Coord(+1, +1, +1), Coord(-1, +1, +1),
  };
  auto integrator = Integrator(coordinate);
  auto scalar_pf = ScalarProjection(integrator);
  scalar_pf.Approximate([](Coord const& xyz){
    return xyz[0] * xyz[1] + xyz[2];
  });
  typename ScalarProjection::Coeff diff
      = scalar_pf.GetCoeffOnTaylorBasis()
      - typename ScalarProjection::Coeff(0, 0, 0, 1, 0, 1, 0, 0, 0, 0);
  EXPECT_NEAR(diff.norm(), 0.0, 1e-14);
  // Check SetZero():
  scalar_pf.SetZero();
  EXPECT_EQ(scalar_pf.coeff(), ScalarProjection::Coeff::Zero());
  // Check SetCoeff():
  typename ScalarProjection::Coeff scalar_coeff
      = ScalarProjection::Coeff::Random();
  EXPECT_NE(scalar_coeff, ScalarProjection::Coeff::Zero());
  scalar_pf.SetCoeff(scalar_coeff);
  EXPECT_EQ(scalar_pf.coeff(), scalar_coeff);
  // Check AddToResidual():
  ScalarProjection::AddToResidual(scalar_pf.coeff(), scalar_coeff.data());
  EXPECT_EQ(scalar_pf.coeff() * 2, scalar_coeff);
  auto vector_pf = VectorProjection(integrator);
  vector_pf.Approximate([](Coord const& xyz)
      -> typename VectorProjection::Value {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    return { 0, 1, x, y, z,
        x * x, x * y, x * z, y * y, y * z, z * z};
  });
  typename VectorProjection::Coeff exact_vector;
  exact_vector.row(0).setZero();
  exact_vector.bottomRows(10).setIdentity();
  typename VectorProjection::Coeff abs_diff
      = vector_pf.GetCoeffOnTaylorBasis() - exact_vector;
  EXPECT_NEAR(abs_diff.norm(), 0.0, 1e-14);
  // Check SetZero():
  vector_pf.SetZero();
  EXPECT_EQ(vector_pf.coeff(), VectorProjection::Coeff::Zero());
  // Check SetCoeff():
  typename VectorProjection::Coeff vector_coeff
      = VectorProjection::Coeff::Random();
  EXPECT_NE(vector_coeff, VectorProjection::Coeff::Zero());
  vector_pf.SetCoeff(vector_coeff);
  EXPECT_EQ(vector_pf.coeff(), vector_coeff);
  // Check AddToResidual():
  VectorProjection::AddToResidual(vector_pf.coeff(), vector_coeff.data());
  EXPECT_EQ(vector_pf.coeff() * 2, vector_coeff);
}

class TestPolynomialHexahedronInterpolation : public TestPolynomialHexahedron {
 protected:
  using Coordinate = mini::coordinate::Hexahedron8<Scalar>;
  // To approximate quadratic functions in each dimension exactly, at least 3 nodes are needed.
  using IntegratorX = mini::integrator::Legendre<Scalar, 3>;
  using IntegratorY = mini::integrator::Lobatto<Scalar, 3>;
  using IntegratorZ = mini::integrator::Lobatto<Scalar, 4>;

  template <bool kLocal>
  static void CheckStaticMembers();

  template <bool kLocal>
  static void CheckValues();

  template <bool kLocal>
  static void CheckDerivatives();

  template <bool kLocal>
  static void CheckCollinearPoints();
};

template <bool kLocal>
void TestPolynomialHexahedronInterpolation::CheckStaticMembers() {
  using Interpolation = mini::polynomial::Hexahedron<
      IntegratorX, IntegratorY, IntegratorZ, kComponents, kLocal>;
  using Integrator = typename Interpolation::Integrator;
  constexpr int N = Interpolation::N;
  constexpr int K = Interpolation::K;
  mini::algebra::Vector<Scalar, N * K> output;
  output.setZero();
  EXPECT_EQ(output.norm(), 0.0);
  Scalar *output_data = output.data();
  for (int i_basis = 0; i_basis < N; ++i_basis) {
    Value value;
    for (int i_comp = 0; i_comp < K; ++i_comp) {
      value[i_comp] = rand_f();
    }
    Scalar *curr_col = output_data + (K * i_basis), *next_col;
    next_col = Interpolation::AddValueTo(value, output_data, i_basis);
    EXPECT_EQ(next_col, curr_col + K);
    EXPECT_NEAR(output.norm(), value.norm(), 1e-15);
    for (int i_comp = 0; i_comp < K; ++i_comp) {
      EXPECT_EQ(value[i_comp], curr_col[i_comp]);
    }
    next_col = Interpolation::MinusValue(value, output_data, i_basis);
    EXPECT_EQ(next_col, curr_col + K);
    EXPECT_EQ(output.norm(), 0.0);
  }
}
TEST_F(TestPolynomialHexahedronInterpolation, StaticMembers) {
  CheckStaticMembers<true>();
  CheckStaticMembers<false>();
}

template <bool kLocal>
void TestPolynomialHexahedronInterpolation::CheckValues() {
  using Interpolation = mini::polynomial::Hexahedron<
      IntegratorX, IntegratorY, IntegratorZ, kComponents, kLocal>;
  using Integrator = typename Interpolation::Integrator;
  // build a hexa-integrator and a Lagrange basis on it
  auto a = 2.0, b = 3.0, c = 4.0;
  auto coordinate = Coordinate {
    Global(-a, -b, -c), Global(+a, -b, -c),
    Global(+a, +b, -c), Global(-a, +b, -c),
    Global(-a, -b, +c), Global(+a, -b, +c),
    Global(+a, +b, +c), Global(-a, +b, +c),
  };
  auto integrator = Integrator(coordinate);
  // build the interpolation
  auto vector_interp = Interpolation(integrator);
  vector_interp.Approximate(GetExactValue);
  // test values on nodes
  for (int ijk = 0; ijk < Interpolation::N; ++ijk) {
    auto &global = vector_interp.integrator().GetGlobal(ijk);
    auto value = GetExactValue(global);
    value -= vector_interp.GlobalToValue(global);
    EXPECT_NEAR(value.norm(), 0, 1e-13);
  }
  // test values on random points
  for (int i_trial = 0; i_trial < kTrials; ++i_trial) {
    auto global = Global{ rand_f(-a, a), rand_f(-b, b), rand_f(-c, c) };
    auto value = GetExactValue(global);
    value -= vector_interp.GlobalToValue(global);
    EXPECT_NEAR(value.norm(), 0, 1e-12);
  }
  // test SetCoeff() and SetValue()
  vector_interp.SetZero();
  EXPECT_EQ(vector_interp.coeff(), Interpolation::Coeff::Zero());
  Interpolation another_interp = vector_interp;
  typename Interpolation::Coeff coeff;
  for (int ijk = 0; ijk < Interpolation::N; ++ijk) {
    Value value = Value::Random();
    coeff.col(ijk) = value;
    EXPECT_NE(vector_interp.coeff(), coeff);
    vector_interp.SetCoeff(ijk, value);
    another_interp.SetValue(ijk, value);
    EXPECT_NEAR((another_interp.GetValue(ijk) - value).norm(),
        0.0, 1e-15);
  }
  EXPECT_EQ(vector_interp.coeff(), coeff);
}
TEST_F(TestPolynomialHexahedronInterpolation, Values) {
  CheckValues<true>();
  CheckValues<false>();
}

template <bool kLocal>
void TestPolynomialHexahedronInterpolation::CheckDerivatives() {
  using Interpolation = mini::polynomial::Hexahedron<
      IntegratorX, IntegratorY, IntegratorZ, kComponents, kLocal>;
  using Integrator = typename Interpolation::Integrator;
  for (int i_trial = 0; i_trial < kTrials; ++i_trial) {
    // build a hexa-integrator and a Lagrange basis on it
    auto a = 20.0 + rand_f(), b = 30.0 + rand_f(), c = 40.0 + rand_f();
    auto coordinate = Coordinate {
        Global(-a, -b, -c), Global(+a, -b, -c),
        Global(+a, +b, -c), Global(-a, +b, -c),
        Global(-a, -b, +c), Global(+a, -b, +c),
        Global(+a, +b, +c), Global(-a, +b, +c),
    };
    auto integrator = Integrator(coordinate);
    // build the interpolation
    coeff_ = Value::Random();
    auto interp = Interpolation(integrator);
    interp.Approximate(GetExactValue);
    // test values and derivatives on nodes
    for (int ijk = 0; ijk < Interpolation::N; ++ijk) {
      Local const &local = interp.integrator().GetLocal(ijk);
      Global const &global = interp.integrator().GetGlobal(ijk);
      auto [value, grad, hess] = interp.GetGlobalValueGradientHessian(ijk);
      EXPECT_NEAR((value - GetExactValue(global)).norm(), 0, 1e-12);
      EXPECT_NEAR((value - interp.GlobalToValue(global)).norm(), 0, 1e-10);
      interp.SetValue(ijk, value);
      EXPECT_EQ(value, interp.GetValue(ijk));
      EXPECT_NEAR((grad - interp.LocalToGlobalGradient(local)).norm(), 0,
          1e-13);
      EXPECT_NEAR((grad - interp.GlobalToGlobalGradient(global)).norm(), 0,
          1e-12);
      // compare with analytical derivatives
      EXPECT_NEAR((GetExactGradient(global) - grad).norm(), 0.0, 1e-8);
      EXPECT_NEAR((GetExactHessian(global) - hess).norm(), 0.0, 1e-8);
      // compare with O(h^2) finite difference derivatives
      auto x = global[X], y = global[Y], z = global[Z], h = 1e-5;
      Value left = interp.GlobalToValue(Global(x - h, y, z));
      Value right = interp.GlobalToValue(Global(x + h, y, z));
      grad.row(X) -= (right - left) / (2 * h);
      left = interp.GlobalToValue(Global(x, y - h, z));
      right = interp.GlobalToValue(Global(x, y + h, z));
      grad.row(Y) -= (right - left) / (2 * h);
      left = interp.GlobalToValue(Global(x, y, z - h));
      right = interp.GlobalToValue(Global(x, y, z + h));
      grad.row(Z) -= (right - left) / (2 * h);
      EXPECT_NEAR(grad.norm(), 0, 1e-6);
      Gradient grad_diff = (
          interp.GlobalToGlobalGradient(Global(x + h, y, z)) -
          interp.GlobalToGlobalGradient(Global(x - h, y, z))
      ) / (2 * h);
      EXPECT_NEAR((hess.row(XX) - grad_diff.row(X)).norm(), 0, 1e-8);
      EXPECT_NEAR((hess.row(XY) - grad_diff.row(Y)).norm(), 0, 1e-8);
      EXPECT_NEAR((hess.row(XZ) - grad_diff.row(Z)).norm(), 0, 1e-8);
      grad_diff = (
          interp.GlobalToGlobalGradient(Global(x, y + h, z)) -
          interp.GlobalToGlobalGradient(Global(x, y - h, z))
      ) / (2 * h);
      EXPECT_NEAR((hess.row(YX) - grad_diff.row(X)).norm(), 0, 1e-8);
      EXPECT_NEAR((hess.row(YY) - grad_diff.row(Y)).norm(), 0, 1e-8);
      EXPECT_NEAR((hess.row(YZ) - grad_diff.row(Z)).norm(), 0, 1e-8);
      grad_diff = (
          interp.GlobalToGlobalGradient(Global(x, y, z + h)) -
          interp.GlobalToGlobalGradient(Global(x, y, z - h))
      ) / (2 * h);
      EXPECT_NEAR((hess.row(ZX) - grad_diff.row(X)).norm(), 0, 1e-8);
      EXPECT_NEAR((hess.row(ZY) - grad_diff.row(Y)).norm(), 0, 1e-8);
      EXPECT_NEAR((hess.row(ZZ) - grad_diff.row(Z)).norm(), 0, 1e-8);
    }
  }
}
TEST_F(TestPolynomialHexahedronInterpolation, Derivatives) {
  CheckDerivatives<true>();
  CheckDerivatives<false>();
}

template <bool kLocal>
void TestPolynomialHexahedronInterpolation::CheckCollinearPoints() {
  using Interpolation = mini::polynomial::Hexahedron<
      IntegratorX, IntegratorY, IntegratorZ, kComponents, kLocal>;
  using Integrator = typename Interpolation::Integrator;
  // build a hexa-integrator and a Coordinate interpolation on it
  auto a = 2.0, b = 3.0, c = 4.0;
  auto cell_coordinate = Coordinate {
    Global(-a, -b, -c), Global(+a, -b, -c),
    Global(+a, +b, -c), Global(-a, +b, -c),
    Global(-a, -b, +c), Global(+a, -b, +c),
    Global(+a, +b, +c), Global(-a, +b, +c),
  };
  auto cell_integrator = Integrator(cell_coordinate);
  auto interp = Interpolation(cell_integrator);
  using CoordinateOnFace = mini::coordinate::Quadrangle4<double, 3>;
  /* test on the x_local == +1 face */{
    auto face_coordinate = CoordinateOnFace {
      Global(+a, -b, -c), Global(+a, +b, -c),
      Global(+a, +b, +c), Global(+a, -b, +c),
    };
    auto face_integrator = mini::integrator::Quadrangle<3, IntegratorY, IntegratorZ>(face_coordinate);
    auto const &face_integrator_ref = face_integrator;
    int i_face = interp.FindFaceId(face_coordinate.center());
    EXPECT_EQ(i_face, 2);
    for (int f = 0; f < face_integrator.CountPoints(); ++f) {
      Global global = face_integrator_ref.GetGlobal(f);
      auto ijk_found = interp.FindCollinearPoints(global, i_face);
      EXPECT_EQ(ijk_found.size(), IntegratorX::Q);
      for (int ijk : ijk_found) {
        auto [i, j, k] = interp.basis().index(ijk);
        auto local = cell_integrator.GetLocal(ijk);
        EXPECT_EQ(local[0], IntegratorX::points[i]);
      }
    }
  }
  /* test on the x_local == -1 face */{
    auto face_coordinate = CoordinateOnFace {
      Global(-a, -b, -c), Global(-a, +b, -c),
      Global(-a, +b, +c), Global(-a, -b, +c),
    };
    auto face_integrator = mini::integrator::Quadrangle<3, IntegratorY, IntegratorZ>(face_coordinate);
    auto const &face_integrator_ref = face_integrator;
    int i_face = interp.FindFaceId(face_coordinate.center());
    EXPECT_EQ(i_face, 4);
    for (int f = 0; f < face_integrator.CountPoints(); ++f) {
      Global global = face_integrator_ref.GetGlobal(f);
      auto ijk_found = interp.FindCollinearPoints(global, i_face);
      EXPECT_EQ(ijk_found.size(), IntegratorX::Q);
      for (int ijk : ijk_found) {
        auto [i, j, k] = interp.basis().index(ijk);
        auto local = cell_integrator.GetLocal(ijk);
        EXPECT_EQ(local[0], IntegratorX::points[i]);
      }
    }
  }
  /* test on the y_local == +1 face */{
    auto face_coordinate = CoordinateOnFace {
      Global(-a, +b, -c), Global(+a, +b, -c),
      Global(+a, +b, +c), Global(-a, +b, +c),
    };
    auto face_integrator = mini::integrator::Quadrangle<3, IntegratorX, IntegratorZ>(face_coordinate);
    auto const &face_integrator_ref = face_integrator;
    int i_face = interp.FindFaceId(face_coordinate.center());
    EXPECT_EQ(i_face, 3);
    for (int f = 0; f < face_integrator.CountPoints(); ++f) {
      Global global = face_integrator_ref.GetGlobal(f);
      auto ijk_found = interp.FindCollinearPoints(global, i_face);
      EXPECT_EQ(ijk_found.size(), IntegratorY::Q);
      for (int ijk : ijk_found) {
        auto [i, j, k] = interp.basis().index(ijk);
        auto local = cell_integrator.GetLocal(ijk);
        EXPECT_EQ(local[1], IntegratorY::points[j]);
      }
    }
  }
  /* test on the y_local == -1 face */{
    auto face_coordinate = CoordinateOnFace {
      Global(-a, -b, -c), Global(+a, -b, -c),
      Global(+a, -b, +c), Global(-a, -b, +c),
    };
    auto face_integrator = mini::integrator::Quadrangle<3, IntegratorX, IntegratorZ>(face_coordinate);
    auto const &face_integrator_ref = face_integrator;
    int i_face = interp.FindFaceId(face_coordinate.center());
    EXPECT_EQ(i_face, 1);
    for (int f = 0; f < face_integrator.CountPoints(); ++f) {
      Global global = face_integrator_ref.GetGlobal(f);
      auto ijk_found = interp.FindCollinearPoints(global, i_face);
      EXPECT_EQ(ijk_found.size(), IntegratorY::Q);
      for (int ijk : ijk_found) {
        auto [i, j, k] = interp.basis().index(ijk);
        auto local = cell_integrator.GetLocal(ijk);
        EXPECT_EQ(local[1], IntegratorY::points[j]);
      }
    }
  }
  /* test on the z_local == +1 face */{
    auto face_coordinate = CoordinateOnFace {
      Global(-a, -b, +c), Global(+a, -b, +c),
      Global(+a, +b, +c), Global(-a, +b, +c),
    };
    auto face_integrator = mini::integrator::Quadrangle<3, IntegratorX, IntegratorY>(face_coordinate);
    auto const &face_integrator_ref = face_integrator;
    int i_face = interp.FindFaceId(face_coordinate.center());
    EXPECT_EQ(i_face, 5);
    for (int f = 0; f < face_integrator.CountPoints(); ++f) {
      Global global = face_integrator_ref.GetGlobal(f);
      auto ijk_found = interp.FindCollinearPoints(global, i_face);
      EXPECT_EQ(ijk_found.size(), IntegratorZ::Q);
      for (int ijk : ijk_found) {
        auto [i, j, k] = interp.basis().index(ijk);
        auto local = cell_integrator.GetLocal(ijk);
        EXPECT_EQ(local[2], IntegratorZ::points[k]);
      }
    }
  }
  /* test on the z_local == -1 face */{
    auto face_coordinate = CoordinateOnFace {
      Global(-a, -b, -c), Global(+a, -b, -c),
      Global(+a, +b, -c), Global(-a, +b, -c),
    };
    auto face_integrator = mini::integrator::Quadrangle<3, IntegratorX, IntegratorY>(face_coordinate);
    auto const &face_integrator_ref = face_integrator;
    int i_face = interp.FindFaceId(face_coordinate.center());
    EXPECT_EQ(i_face, 0);
    for (int f = 0; f < face_integrator.CountPoints(); ++f) {
      Global global = face_integrator_ref.GetGlobal(f);
      auto ijk_found = interp.FindCollinearPoints(global, i_face);
      EXPECT_EQ(ijk_found.size(), IntegratorZ::Q);
      for (int ijk : ijk_found) {
        auto [i, j, k] = interp.basis().index(ijk);
        auto local = cell_integrator.GetLocal(ijk);
        EXPECT_EQ(local[2], IntegratorZ::points[k]);
      }
    }
  }
}
TEST_F(TestPolynomialHexahedronInterpolation, CollinearPoints) {
  CheckCollinearPoints<true>();
  CheckCollinearPoints<false>();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
