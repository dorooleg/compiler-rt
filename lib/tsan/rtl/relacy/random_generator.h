#ifndef TSAN_RANDOM_GENERATOR_H
#define TSAN_RANDOM_GENERATOR_H

#include <random>
#include <functional>
#include <vector>
#include <sys/mman.h>

namespace __tsan {

class RandomGenerator {
public:
  RandomGenerator()
      : count_calls_(*(int*)mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, 0, 0))  {
    generator_ = 0;
    count_calls_ = 0;
    generators_ = {
        [&](int b) {
          static std::mt19937 gen(rd());
          std::uniform_int_distribution<> dis(0, b);
          return abs(static_cast<int>(dis(gen)) % b);
        },
        [&](int b) {
          static std::mt19937 gen(rd());
          std::binomial_distribution<> dis(b, 0.5);
          return abs(static_cast<int>(dis(gen)) % b);
        },
        [&](int b) {
          static std::mt19937 gen(rd());
          std::negative_binomial_distribution<> dis(b, 0.75);
          return abs(static_cast<int>(dis(gen)) % b);
        },
        [&](int b) {
          static std::mt19937 gen(rd());
          std::geometric_distribution<> dis;
          return abs(static_cast<int>(dis(gen)) % b);
        },
        [&](int b) {
          return abs(static_cast<int>(rd()) % b);
        },
        [&](int b) {
          static std::mt19937 gen(rd());
          std::poisson_distribution<> dis(4);
          return abs(static_cast<int>(dis(gen)) % b);
        },
        [&](int b) {
          static std::mt19937 gen(rd());
          std::exponential_distribution<> dis(1);
          return abs(static_cast<int>(dis(gen)) % b);
        },
        [&](int b) {
          static std::mt19937 gen(rd());
          std::weibull_distribution<> dis;
          return abs(static_cast<int>(dis(gen)) % b);
        },
        [&](int b) {
          static std::mt19937 gen(rd());
          std::normal_distribution<> dis(b, 2);
          return abs(static_cast<int>(dis(gen)) % b);
        },
        [&](int b) {
          static std::mt19937 gen(rd());
          std::lognormal_distribution<> dis(b, 0.25);
          return abs(static_cast<int>(dis(gen)) % b);
        }
    };

  }

  int Rand(int b) {
    ++count_calls_;
    return generators_[generator_ % generators_.size()](b);
  }

  void NextGenerator() {
    for (int i = 0; i < count_calls_; i++) {
      generators_[generator_ % generators_.size()](10);
    }
    count_calls_ = 0;
    ++generator_;
  }

private:
  int &count_calls_;
  int generator_;
  std::random_device rd;
  std::vector<std::function<int (int)>> generators_;
};

}

#endif //TSAN_RANDOM_GENERATOR_H
