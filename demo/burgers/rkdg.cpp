//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include "rkdg.hpp"

namespace mini {
namespace mesh {

template class Shuffler<idx_t, double>;

namespace cgns {

template class Part<cgsize_t, double, 1, 3, 0>;
template class Part<cgsize_t, double, 1, 3, 2>;

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

namespace mini {
namespace riemann {
namespace rotated {

template class Single<3>;
template class Burgers<3>;

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

namespace mini {
namespace polynomial {

template class mini::polynomial::LazyWeno<Cell0>;
template class mini::polynomial::LazyWeno<Cell2>;

}  // namespace polynomial
}  // namespace mini

template class RungeKutta<1, mini::mesh::cgns::Part0,
    mini::riemann::rotated::Burgers<3>, mini::polynomial::Limiter0>;
template class RungeKutta<3, mini::mesh::cgns::Part2,
    mini::riemann::rotated::Burgers<3>, mini::polynomial::Limiter2>;