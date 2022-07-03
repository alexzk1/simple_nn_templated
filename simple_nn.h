#pragma once

#include <tuple>
#include <random>
#include <cmath>
#include <execution>
#include <functional>

#include "types_helpers.h"
#include "cm_ctors.h"
#include "matrix2d.h"

///should be at least 2 numbers passed - input and output layer,
///more numbers between are sizes of hidden layers
template <class Float, size_t ...Args>
class SimpleLayeredNN
{
public:
    static constexpr size_t layers_count = sizeof...(Args);

    template<size_t R, size_t C>
    using WeightsMatrixT = Matrix2D<Float, R, C>;

    constexpr static size_t inputs_count  = thelpers::first_v<Args...>();
    constexpr static size_t outputs_count = thelpers::last_v<Args...>();
private:
    static_assert(std::is_floating_point<Float>::value, "Expecting floating point type only.");
    static_assert(layers_count > 1, "Expecting at least 2 additional template parameters.");

    //builds tuple of weight matrixes recursively out of template sizes
    template <size_t index, class Tuple>
    static auto make_weights_reccur(Tuple&& src) noexcept
    {
        static constexpr bool is        = index < layers_count - 1u;
        static constexpr auto src_tuple = std::make_tuple(Args...);
        static constexpr auto size_frst = thelpers::pop_back(src_tuple);
        static constexpr auto size_next = thelpers::pop_front(src_tuple);

        if constexpr (!is)
            return src;

        if constexpr(is)
        {
            //rows columns are swapped, see book why, rows are next layer and columns current layer
            auto tmp = std::make_tuple(WeightsMatrixT<std::get<index>(size_next), std::get<index>(size_frst)>());
            return make_weights_reccur<index+1>(std::move(std::tuple_cat(src, tmp)));
        }
    }

    //builds all weight matrices
    static auto make_weights() noexcept
    {
        return make_weights_reccur<0>(std::move(std::tuple{}));
    }

    //fills single weight matrix with random values, gaussian distribution where
    //stddev of it is 1 / root(incoming_connections)
    template <size_t R, size_t C>
    static void fill_matrix_random(WeightsMatrixT<R, C>& src) noexcept
    {
        using engine_t = std::conditional< (7 < sizeof(void*)), std::mt19937_64, std::mt19937>::type;

        //seeding pseudo random engine by real entropy
        engine_t pseudo_rnd(std::random_device{}());

        //making distributor
        const Float sdev = pow(cast(src.rows()), cast(-0.5f));
        std::normal_distribution<Float> dis(cast(0.), sdev);

        const auto rnd =[&dis, &pseudo_rnd]()
        {
            Float v, a;
            do
            {
                v = dis(pseudo_rnd);
                a = std::fabs(v);
            }
            while(a < 0.001 && a > 0.999);
            return v;
        };

        for (auto& v : src)
            v = rnd();
    }

    //recursively applies random for each matrix in tuple
    template <class T, class ...Ts>
    static void fill_random_1by1(T& left, Ts& ...others) noexcept
    {
        fill_matrix_random(left);
        if constexpr (sizeof...(Ts) > 0)
        {
            fill_random_1by1(others...);
        }
    }

//----------------------------------------------------------------------------------
#define NO_COPY_PASTE(NAME) if constexpr (szo < 1) \
{ \
    if constexpr (!KeepAll) \
        return o; \
    if constexpr (KeepAll) \
        return std::make_tuple(o); \
} \
if constexpr (szo > 0) \
{ \
    if constexpr (KeepAll)\
        return std::tuple_cat(std::make_tuple(o),  forward<KeepAll>(o, others...));\
    if constexpr (!KeepAll)\
        return NAME<KeepAll>(o, others...);}
//----------------------------------------------------------------------------------

    //if KeepAll = true then it will return all calculations as tuple
    //otherwise it will return only last one as single value
    template <bool KeepAll, class Inps, class T, class ...Ts>
    static decltype(auto) forward(const Inps& inps, T& left, Ts& ...others) noexcept
    {
        constexpr auto szo = sizeof...(others);
        const auto o = activation_function(left.dot(inps));
        NO_COPY_PASTE(forward);
    }

    template <bool KeepAll, class Inps, class T, class ...Ts>
    static decltype(auto) backward(const Inps& inps, T& left, Ts& ...others) noexcept
    {
        constexpr auto szo = sizeof...(others);
        const auto o = reverse_activation_function(left.dot(inps));
        NO_COPY_PASTE(backward);
    }
#undef NO_COPY_PASTE

    template <class Mat>
    static Mat activation_function(const Mat& src) noexcept
    {
        constexpr static Float one  = cast(1.f);
        Mat res;
        //FIXME: doing parallel here shows data-race by thread sanitizer, not sure why yet...
        std::transform(std::execution::par_unseq, src.begin(), src.end(), res.begin(), [](const Float& x)
        {
            return one / (one + cast(exp(-x)));
        });
        return res;
    }

    template <class Mat>
    static Mat reverse_activation_function(const Mat& src) noexcept
    {
        constexpr static Float one  = cast(1.f);
        Mat res;
        std::transform(std::execution::par_unseq, src.begin(), src.end(), res.begin(), [](const Float& y)->Float
        {
            return log1p(static_cast<Float>(y / (one - y)));
        });
        return res;
    }

    template <size_t Index, class IniErrs, class ...Tw>
    static auto build_errors(IniErrs&& errs, const std::tuple<Tw...>& w)
    {
        constexpr auto max_size = sizeof...(Tw) - 1;
        constexpr bool keep_recurse = Index < max_size;

        if constexpr (keep_recurse)
        {
            auto newerr = std::make_tuple(std::get<Index>(w).transpose().dot(std::get<Index>(errs)));
            return build_errors<Index+1>(std::tuple_cat(std::move(errs), std::move(newerr)), w);
        }

        if constexpr (!keep_recurse)
            return errs;
    }

    template <size_t Index, class Errors, class Outs, class ...Tw>
    static void update_weights(const Float learning_rate, const Errors& err, const Outs& outs, std::tuple<Tw...>& w)
    {
        if constexpr(Index < sizeof...(Tw))
        {
            {
                //separated block, so extra data are deallocated prior recursive call
                const auto& o  = std::get<Index>(outs);
                const auto no  = std::get<Index + 1>(outs).transpose();
                const auto m1  = std::get<Index>(err) * o * (cast(1) - o);
                std::get<Index>(w) += m1.dot(no) * learning_rate;
            }
            update_weights<Index + 1>(learning_rate, err, outs, w);
        }
    }
private:
    std::invoke_result_t<decltype(&make_weights)> weights{make_weights()};
public:
    SimpleLayeredNN() = default;
    ~SimpleLayeredNN()= default;
    DEFAULT_COPYMOVE(SimpleLayeredNN);

    ///set all weights randomly
    SimpleLayeredNN& random_weights() noexcept
    {
        std::apply([](auto& a, auto& ... b)
        {
            fill_random_1by1(a, b...);
        }, weights);

        return *this;
    }

    template <bool KeepAllOuts = false>
    auto query(const VectorRow<Float, inputs_count>& inputs) const noexcept
    {
        return std::apply([&](auto& a, auto& ... b)
        {
            return forward<KeepAllOuts>(inputs, a, b...);
        }, weights);
    }


    template <bool KeepAllOuts = false>
    auto reverse_query(const VectorRow<Float, outputs_count>& outputs) const noexcept
    {
        const auto rweights = thelpers::reverse_tuple_ref(weights);
        return std::apply([&outputs](auto& a, auto& ... b)
        {
            return backward<KeepAllOuts>(outputs, a, b...);
        }, rweights);
    }


    void train(const Float learning_rate, const VectorRow<Float, inputs_count>& inputs, const VectorRow<Float, outputs_count>& targets)
    {
        //outputs from each layer
        const auto outputs = query<true>(inputs);
        {
            //tuples of references to matrices in reverse order
            const auto routputs = std::tuple_cat(thelpers::reverse_tuple_ref(outputs), std::tie(inputs));
            auto rweights = thelpers::reverse_tuple_ref(weights);
            const auto errors   = build_errors<0>(std::make_tuple(targets - std::get<0>(routputs)), rweights);

            update_weights<0>(learning_rate, errors, routputs, rweights);
        }
    }

    ///alias for static_cast<Float> template parameter
    template<class Any>
    static constexpr Float cast(const Any v) noexcept
    {
        return static_cast<Float>(v);
    }
};

