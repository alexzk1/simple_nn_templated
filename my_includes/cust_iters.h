#pragma once
#include <type_traits>
#include <iterator>
#include "cm_ctors.h"

template <class I>
class count_iter
{
protected:
    I i{0};
public:
    static_assert(std::is_integral<I>::value, "Integer type required.");
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = I;
    using difference_type = std::ptrdiff_t;
    using reference = I;
    using pointer = I;

    count_iter()  noexcept = default;
    ~count_iter() noexcept = default;

    explicit count_iter(I i) noexcept :i(i) {}
    DEFAULT_COPYMOVE(count_iter);

    reference operator*() const noexcept
    {
        return i;
    }

    count_iter& operator++() noexcept
    {
        ++i;
        return *this;
    }

    count_iter operator++(int) noexcept
    {
        return count_iter(i++);
    }

    count_iter& operator--() noexcept
    {
        --i;
        return *this;
    }

    count_iter operator--(int) noexcept
    {
        return count_iter(i--);
    }

    count_iter& operator += (const difference_type& n) noexcept
    {
        i += n;
        return *this;
    }

    count_iter& operator -= (const difference_type& n) noexcept
    {
        i -= n;
        return *this;
    }

    count_iter operator + (const difference_type& n) const noexcept
    {
        return count_iter(i + n);
    }

    count_iter operator - (const difference_type& n) const noexcept
    {
        return count_iter(i - n);
    }

    const I& base() const noexcept
    {
        return i;
    }

    bool operator ==(const count_iter& c) const noexcept
    {
        return i == c.i;
    }

    bool operator !=(const count_iter& c) const noexcept
    {
        return i != c.i;
    }
};

using IndexIter = count_iter<size_t>;
