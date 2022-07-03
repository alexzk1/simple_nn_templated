#pragma once
#include <cstddef>
#include <numeric>
#include <stdint.h>
#include <array>
#include <iostream>
#include <algorithm>
#include <execution>
#include <initializer_list>

#include "cm_ctors.h"
#include "cust_iters.h"
#include "palign.h"
#include "types_helpers.h"

//destructive size allows different elements be separated for different threads
//https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0154r1.html

#define MATRIX_ALIGN prefFloatsAlign()

template<typename T, std::size_t ALIGNMENT_IN_BYTES>
using AlignedVector = std::vector<T, AlignedAllocator<T, ALIGNMENT_IN_BYTES> >;

template <typename Tp, size_t Rows, size_t Cols>
class Matrix2D;

template <typename Tp, size_t Cols>
using VectorCol = Matrix2D<Tp, 1, Cols>;

template <typename Tp, size_t Rows>
using VectorRow = Matrix2D<Tp, Rows, 1>;


template <typename Tp, size_t Rows, size_t Cols>
class Matrix2D
{
private:
    static_assert(std::is_arithmetic<Tp>::value, "Only numbers are supported.");
    AlignedVector<Tp, MATRIX_ALIGN> data;

    void resize()
    {
        data.resize(Cols * Rows);
    }

    static constexpr size_t index(const size_t r, const size_t c) noexcept
    {
        return r * Cols + c;
    }

    void init(const std::initializer_list<const std::initializer_list<Tp>>& src)
    {
        if (src.size() != Rows)
            throw std::range_error("Wrong rows amount in initializer list.");

        resize();
        for (size_t r = 0; r < Rows; ++r)
        {
            const auto &ref = src.begin()[r];
            if (ref.size() != Cols)
                throw std::range_error("Wrong colss amount in initializer list.");
            std::copy(std::execution::par_unseq, ref.begin(), ref.end(), begin() + index(r, 0));
        }
    }

    void init(const std::initializer_list<Tp>& src)
    {
        static_assert(is_vector(), "Only vectors are supported by 1D list.");
        if (src.size() != Cols * Rows)
            throw std::range_error("Wrong size for vector in initializer list.");

        resize();
        std::copy(std::execution::par_unseq, src.begin(), src.end(), data.begin());
    }
public:

    static constexpr bool is_vector() noexcept
    {
        return Rows == 1 || Cols == 1;
    }

    constexpr auto rows() const noexcept
    {
        return Rows;
    }

    constexpr auto cols () const noexcept
    {
        return Cols;
    }

    size_t size() const noexcept
    {
        return data.size();
    }

    Matrix2D()
    {
        resize();
    }
    ~Matrix2D() = default;
    DEFAULT_COPYMOVE(Matrix2D);

    Matrix2D(const std::initializer_list<const std::initializer_list<Tp>>& src)
    {
        init(src);
    }

    Matrix2D(const std::initializer_list<Tp>& src)
    {
        init(src);
    }

    auto& operator=(const std::initializer_list<const std::initializer_list<Tp>>& src)
    {
        init(src);
        return *this;
    }

    auto& operator = (const std::initializer_list<Tp>& src)
    {
        init(src);
        return *this;
    }

    const Tp& at(const size_t r, const size_t c) const
#ifdef NDEBUG
    noexcept
#endif
    {
#ifdef NDEBUG
        return *(begin() + index(r,c));
#else
        return data.at(index(r, c));
#endif
    }

    Tp& at(const size_t r, const size_t c)
#ifdef NDEBUG
    noexcept
#endif
    {
#ifdef NDEBUG
        return *(begin() + index(r,c));
#else
        return data.at(index(r, c));
#endif
    }

    template <size_t cls>
    auto dot(const Matrix2D<Tp, Cols, cls> &by) const
    {
        Matrix2D<Tp, Rows, cls> res;
        constexpr bool p1 = Rows > 1;
        constexpr bool p2 = !p1 && cls > 1;

        std::for_each(thelpers::seq_or_par<p1>::value(), IndexIter(0), IndexIter(Rows), [&](auto r)
        {
            std::for_each(thelpers::seq_or_par<p2>::value(), IndexIter(0), IndexIter(cls), [&](auto cres)
            {
                constexpr auto zero = static_cast<Tp>(0);
                const auto beg = begin() + index (r, 0);
                res.at(r, cres) = std::accumulate(IndexIter(0), IndexIter(Cols), zero, [&](const auto old, const auto k)
                {
                    return old + *(beg + k) * by.at(k, cres);
                });
            });
        });
        return res;
    }

    Matrix2D<Tp, Cols, Rows> transpose() const
    {
        Matrix2D<Tp, Cols, Rows> res;
        for (size_t i = 0; i < Cols; ++i)
            for (size_t j = 0; j < Rows; ++j)
                res.at(i, j) = at(j, i);
        return res;
    }

    void set_zero()
    {
        std::fill(std::execution::par_unseq, begin(), end(), Tp(0));
    }

    //element-to-element arithmetic of the same-sized matrices
    auto& operator *=(const Matrix2D<Tp, Rows, Cols>& c)
    {
        std::for_each(std::execution::par_unseq, IndexIter(0), IndexIter(Rows * Cols), [&](auto i)
        {
            *(begin() + i) *= *(c.begin() + i);
        });
        return *this;
    }

    auto operator *(const Matrix2D<Tp, Rows, Cols>& c) const
    {
        Matrix2D<Tp, Rows, Cols> res(*this);
        res *= c;
        return res;
    }

    auto& operator /=(const Matrix2D<Tp, Rows, Cols>& c)
    {
        std::for_each(std::execution::par_unseq, IndexIter(0), IndexIter(Rows * Cols), [&](auto i)
        {
            *(begin() + i) /= *(c.begin() + i);
        });
        return *this;
    }

    auto operator /(const Matrix2D<Tp, Rows, Cols>& c) const
    {
        Matrix2D<Tp, Rows, Cols> res(*this);
        res /= c;
        return res;
    }

    auto& operator +=(const Matrix2D<Tp, Rows, Cols>& c)
    {
        std::for_each(std::execution::par_unseq, IndexIter(0), IndexIter(Rows * Cols), [&](auto i)
        {
            *(begin() + i) += *(c.begin() + i);
        });
        return *this;
    }

    auto operator +(const Matrix2D<Tp, Rows, Cols>& c) const
    {
        Matrix2D<Tp, Rows, Cols> res(*this);
        res += c;
        return res;
    }

    auto& operator -=(const Matrix2D<Tp, Rows, Cols>& c)
    {
        std::for_each(std::execution::par_unseq, IndexIter(0), IndexIter(Rows * Cols), [&](auto i)
        {
            *(begin() + i) -= *(c.begin() + i);
        });
        return *this;
    }

    auto operator -(const Matrix2D<Tp, Rows, Cols>& c) const
    {
        Matrix2D<Tp, Rows, Cols> res(*this);
        res -= c;
        return res;
    }

    //matrix by scalar arithmetics
    auto& operator *=(const Tp v)
    {
        std::transform(std::execution::par_unseq, begin(), end(), begin(), [&v](const auto n)
        {
            return n * v;
        });
        return *this;
    }

    auto operator *(const Tp v) const
    {
        Matrix2D<Tp, Rows, Cols> res(*this);
        res *= v;
        return res;
    }

    auto& operator /=(const Tp v)
    {
        std::transform(std::execution::par_unseq, begin(), end(), begin(), [&v](const auto n)
        {
            return n / v;
        });
        return *this;
    }

    auto operator /(const Tp v) const
    {
        Matrix2D<Tp, Rows, Cols> res(*this);
        res /= v;
        return res;
    }

    auto& operator +=(const Tp v)
    {
        std::transform(std::execution::par_unseq, begin(), end(), begin(), [&v](const auto n)
        {
            return n + v;
        });
        return *this;
    }

    auto operator +(const Tp v) const
    {
        Matrix2D<Tp, Rows, Cols> res(*this);
        res += v;
        return res;
    }

    auto& operator -=(const Tp v)
    {
        std::transform(std::execution::par_unseq, begin(), end(), begin(), [&v](const auto n)
        {
            return n - v;
        });
        return *this;
    }

    auto operator -(const Tp v) const
    {
        Matrix2D<Tp, Rows, Cols> res(*this);
        res -= v;
        return res;
    }

    auto begin() noexcept
    {
        return data.begin();
    }

    auto end() noexcept
    {
        return data.end();
    }

    auto begin() const noexcept
    {
        return data.cbegin();
    }

    auto end() const noexcept
    {
        return data.cend();
    }

    template <class OS>
    void dump(OS& s) const
    {
        s << "{";
        for (size_t r = 0; r < rows(); ++r)
        {
            s << " {";
            for(size_t c = 0; c < cols(); ++c)
                s << data[index(r,c)] << ",";
            s << " }," << std::endl;
        }
        s << "}" << std::endl;
    }
};

template <class T, size_t R, size_t C>
inline std::ostream& operator << (std::ostream &s, const Matrix2D<T, R, C>& mat)
{
    mat.dump(s);
    return s;
}


namespace mtest
{
    inline auto test1()
    {
        static Matrix2D<float, 2, 3> a =
        {
            {0, 4, -2},
            {-4, -3, 0},
        };
        static Matrix2D<float, 3, 2> b=
        {
            {0, 1},
            {1, -1},
            {2, 3},
        };
        /*
        Expecting result:
        {
          {0, -10},
          {-3, -1}
        }
        */
        return a.dot(b);
    }

    inline auto test2()
    {
        static Matrix2D<float, 2, 3> a =
        {
            {0, 4, -2},
            {-4, -3, 0},
        };
        static Matrix2D<float, 3, 1> b=
        {
            0,
            1,
            2,
        };
        /*
        Expecting result:
        {
          {0,},
          {-3},
        }
        */
        return a.dot(b);
    }
}

///allows to do like 1 - matrix
template <class Float, size_t Rows, size_t Cols>
inline Matrix2D<Float, Rows, Cols> operator - (const Float v, Matrix2D<Float, Rows, Cols> src)
{
    std::transform(std::execution::par_unseq, std::begin(src), std::end(src), std::begin(src),
                   [&v](const Float x)
    {
        return v - x;
    });
    return src;
}

#undef MATRIX_ALIGN
