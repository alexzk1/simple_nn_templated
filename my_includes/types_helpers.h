#pragma once
#include <tuple>
#include <type_traits>
#include <cstddef>
#include <execution>

namespace thelpers
{
    //https://stackoverflow.com/questions/20045617/check-for-arguments-type-in-a-variadic-template-declaration
    template <typename ...>
    struct are_same : std::true_type {};

    template <typename S, typename T, typename ... Ts>
    struct are_same <S, T, Ts...> : std::false_type {};

    template <typename T, typename ... Ts>
    struct are_same <T, T, Ts...> : are_same<T, Ts...> {};

    template <typename ... Ts>
    inline constexpr bool are_same_v = are_same<Ts...>::value;

    template<typename x_Tuple, std::size_t... x_index>
    inline constexpr decltype(auto) make_tuple_helper(const x_Tuple& other, ::std::index_sequence<x_index...>)
    {
        return std::make_tuple(std::get<x_index>(other)...);
    }

    template <typename Tuple>
    constexpr decltype(auto) pop_front(const Tuple& tuple)
    {
        static_assert(std::tuple_size<Tuple>::value > 0, "Cannot pop from empty tuple.");
        return std::apply([](auto, auto...rest)
        {
            return std::make_tuple(rest...);
        }, tuple);
    }

    template<typename... x_Field>
    inline constexpr decltype(auto) pop_back(const std::tuple<x_Field...>& other)
    {
        static_assert(sizeof...(x_Field) > 0, "Cannot pop from empty tuple.");
        return make_tuple_helper(other, std::make_index_sequence<sizeof...(x_Field) - std::size_t{1u}> {});
    }

    template <size_t First, size_t ...>
    constexpr inline auto first_v()
    {
        return First;
    }

    template <size_t First, size_t ...Args>
    constexpr inline auto last_v()
    {
        constexpr bool sz = sizeof...(Args) > 0;
        if constexpr (sz)
            return last_v<Args...>();
        if constexpr (!sz)
            return First;
    }

    template <class T, size_t ...I>
    auto reverse_impl_ref(T&& t, std::index_sequence<I...>)
    {
        //using std::forward_as_tuple will keep original l/r reference
        return std::forward_as_tuple(std::get<sizeof...(I) - 1 - I>(std::forward<T>(t))...);
    }

    template<class ...Ts>
    auto reverse_tuple_ref(const std::tuple<Ts...>& t)
    {
        //as we have reference t here, reverse_impl_ref will keep it
        //so result of this call is tuple of REFERENCES to original elements
        //make sure source tuple lives longer
        return reverse_impl_ref(t, std::make_index_sequence<sizeof...(Ts)>());
    }

    template<class ...Ts>
    auto reverse_tuple_ref(std::tuple<Ts...>& t)
    {
        //as we have reference t here, reverse_impl_ref will keep it
        //so result of this call is tuple of REFERENCES to original elements
        //make sure source tuple lives longer
        return reverse_impl_ref(t, std::make_index_sequence<sizeof...(Ts)>());
    }

    template <class ...Ts>
    constexpr auto size(const std::tuple<Ts...>&)
    {
        return sizeof...(Ts);
    }

    template <bool Parallel>
    struct seq_or_par
    {
        using type = typename std::conditional<Parallel, std::execution::parallel_policy,
              std::execution::sequenced_policy>::type;
        static constexpr auto& value()
        {
            if constexpr(!Parallel)
                return std::execution::seq;

            if constexpr(Parallel)
                return std::execution::par;
        }
    };
}
