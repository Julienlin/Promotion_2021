#ifndef __UTILS_H_
#define __UTILS_H_

#include <vector>

namespace utils {
template <typename T> T avg(std::vector<T> &vect, int nb) {
  T avg;
  for (int i = 0; i < nb; i++) {
        avg += vect[i];
  }
  return avg / nb;
}
} // namespace utils

#endif // __UTILS_H_
