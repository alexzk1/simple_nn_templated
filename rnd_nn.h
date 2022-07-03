#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <type_traits>
#include <random>
#include <tuple>

namespace rnd_nn
{
    template <class Float = float>
    inline Float gen() noexcept
    {
        using engine_t = std::conditional< (7 < sizeof(void*)), std::mt19937_64, std::mt19937>::type;
        static_assert(std::is_floating_point<Float>::value, "Floating point type expected.");

        //seeding pseudo random engine by real entropy
        static engine_t pseudo_rnd(std::random_device{}());

        //making distributor in range [0.1; 0.7) as NN does not like 0 and 1
        static std::uniform_real_distribution<Float> dis(static_cast<Float>(0.1), static_cast<Float>(0.7));
        return dis(pseudo_rnd);
    }

    template <class Float, int ...Ts>
    inline void fill_random(Eigen::Matrix<Float, Ts...>& src) noexcept
    {
        const size_t sz = src.cols() * src.rows();
        for (size_t i = 0; i < sz; ++i)
            *(src.data() + i) = gen<Float>();

    }

    template <class Float, class T, class ...Ts>
    void fill_random_1by1(T& left, Ts& ...others)
    {
        fill_random<Float>(left);
        if constexpr (sizeof...(Ts) > 0)
        {
            fill_random_1by1<Float>(others...);
        }
    }

    template <class Float, class ...Ts>
    inline void fill_random(std::tuple<Ts...> &src)
    {
        std::apply([](auto& a, auto& ... b)
        {
            fill_random_1by1<Float>(a, b...);
        }, src);
    }
}
