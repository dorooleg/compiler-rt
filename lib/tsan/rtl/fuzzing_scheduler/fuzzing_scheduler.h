#pragma once

namespace __tsan {

struct IFuzzingScheduler
{
    virtual void SynchronizationPoint() = 0;
};

IFuzzingScheduler& GetFuzzingScheduler();

}  // namespace __tsan
