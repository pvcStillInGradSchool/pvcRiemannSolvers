#ifndef MINI_RAND_HPP_
#define MINI_RAND_HPP_

#include <cstdlib>

namespace mini {
namespace rand {

/**
 * @brief Uniformly pick a real value from the range [​a​, b). 
 * 
 * @param a 
 * @param b 
 * @return double 
 */
inline double uniform(double a, double b) {
  return a + (b - a) * std::rand() / (1.0 + RAND_MAX);
}

/**
 * @brief The reentrant version of `mini::rand::uniform()`.
 * 
 * @param a 
 * @param b 
 * @param seedp 
 * @return double 
 */
inline double uniform_r(double a, double b, unsigned int *seedp) {
  return a + (b - a) * rand_r(seedp) / (1.0 + RAND_MAX);
}

}  // namespace rand
}  // namespace mini

#endif  // MINI_RAND_HPP_
