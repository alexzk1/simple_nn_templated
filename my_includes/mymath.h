#pragma once
#include <type_traits>
#include <cmath>
#include <cstdint>

namespace math
{
    template <class T>
    constexpr int sgn(const T v)
    {
        static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported.");
        static_assert(static_cast<int>(true) == 1 && static_cast<int>(false) == 0, "Woops!");

        if constexpr(std::is_signed<T>::value)
            return static_cast<int>(v > T(0)) - static_cast<int>(v < T(0));

        if constexpr(std::is_unsigned<T>::value)
            return static_cast<int>(v > T(0));
    }

    template <class T>
    constexpr T abs(const T x)
    {
        static_assert(std::is_arithmetic<T>::value, "Only arithmetic types are supported.");

        if constexpr (std::is_integral<T>::value)
            return std::abs(x);

        if constexpr (std::is_floating_point<T>::value)
            return std::fabs(x);
    }
}
