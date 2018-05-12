#ifndef TSAN_RANDOM_GENERATOR_H
#define TSAN_RANDOM_GENERATOR_H

#include <random>
#include <functional>
#include <vector>
#include "rtl/relacy/tsan_shared_value.h"

namespace __tsan {
namespace __relacy {

class RandomGenerator {
  public:
   RandomGenerator();

   int Rand(int b);

   void NextGenerator();

  private:
   SharedValue<int> count_calls_;
   int generator_;
   std::random_device rd;
   std::vector<std::function<int(int)>> generators_;
};

}
}

#endif //TSAN_RANDOM_GENERATOR_H
