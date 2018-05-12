#include "rtl/relacy/schedulers/tsan_random_generator.h"

namespace __tsan {
namespace __relacy {

RandomGenerator::RandomGenerator() {
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

int RandomGenerator::Rand(int b) {
    ++count_calls_;
    return generators_[generator_ % generators_.size()](b);
}

void RandomGenerator::NextGenerator() {
    for (int i = 0; i < count_calls_; i++) {
        generators_[generator_ % generators_.size()](1);
    }
    count_calls_ = 0;
    ++generator_;
}

}
}