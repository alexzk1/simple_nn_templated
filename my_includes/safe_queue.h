// Created by Adam Kecskes
// https://github.com/K-Adam/SafeQueue

#pragma once

#include <mutex>
#include <condition_variable>

#include <queue>
#include <utility>
#include "cm_ctors.h"

template<class T>
class SafeQueue
{
private:
    std::queue<T> q;
    mutable std::mutex mtx;
    std::condition_variable cv;
    std::condition_variable sync_wait;
    bool finish_processing = false;
    int sync_counter = 0;

    void decSyncCounter()
    {
        if (--sync_counter == 0)
        {
            sync_wait.notify_one();
        }
    }

public:
    using size_type  = typename std::queue<T>::size_type;
    using value_type = T;

    SafeQueue() = default;
    DEFAULT_COPYMOVE(SafeQueue);

    ~SafeQueue()
    {
        finishSync();
    }

    void push(T&& item)
    {
        std::lock_guard<std::mutex> lock(mtx);
        q.push(std::move(item));
        cv.notify_one();
    }

    template <class ...Args>
    void emplace(Args ...args)
    {
        push(T(std::forward<Args>(args)...));
    }

    size_type size() const
    {
        std::lock_guard<std::mutex> lock(mtx);
        return q.size();
    }

    [[nodiscard]] bool pop(T& item)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (q.empty())
        {
            return false;
        }
        item = std::move(q.front());
        q.pop();
        return true;
    }

    [[nodiscard]]
    bool popSync(T& item)
    {
        std::unique_lock<std::mutex> lock(mtx);
        ++sync_counter;

        cv.wait(lock, [this]
        {
            return !q.empty() || finish_processing;
        });

        bool res;
        if ((res = !q.empty()))
        {
            item = std::move(q.front());
            q.pop();
        }
        decSyncCounter();
        return res;
    }

    void finishSync()
    {
        std::unique_lock<std::mutex> lock(mtx);
        finish_processing = true;
        cv.notify_all();

        sync_wait.wait(lock, [this]()
        {
            return sync_counter == 0;
        });
        finish_processing = false;
    }
};
