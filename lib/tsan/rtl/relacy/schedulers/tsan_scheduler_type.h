#ifndef TSAN_SCHEDULER_TYPE_H
#define TSAN_SCHEDULER_TYPE_H


namespace __tsan {
namespace __relacy {

enum class SchedulerType {
   OS,
   ALL_STATES,
   FIXED_WINDOW,
   FULL_PATH,
   PARALLEL_FULL_PATH,
   RANDOM,
   RANDOM_WITH_DIFFERENT_DISTRIBUTIONS
};

}
}

#endif //TSAN_SCHEDULER_TYPE_H
