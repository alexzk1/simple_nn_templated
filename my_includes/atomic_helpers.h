#pragma once
#include <atomic>

namespace atomics
{
    //tests atomic bool for expected, if it is - sets it to !expected and returns true
    inline bool testandflip(std::atomic<bool>& var, const bool expected)
    {
        bool exp{expected};
        return var.compare_exchange_strong(exp, !expected);
    }

    //atomic var = var || value
    inline void or_equal(std::atomic<bool>& var, const bool value)
    {
        /*
         * Compares the contents of the contained value with expected:
            - if true, it replaces the contained value with val (like store).
            - if false, it replaces expected with the contained value .
         * */
        bool expected = false;
        var.compare_exchange_strong(expected, value);
    }

    //atomic var = var && value
    inline void and_equal(std::atomic<bool>& var, const bool value)
    {
        /*
         * Compares the contents of the contained value with expected:
            - if true, it replaces the contained value with val (like store).
            - if false, it replaces expected with the contained value .
         * */
        bool expected = true;
        var.compare_exchange_strong(expected, value);
    }
}
