#pragma once

#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "cm_ctors.h"

//warning, this is super-simple parser which does not handle quatter, embedded comas etc
namespace csv
{
    class row
    {
    public:
        DEFAULT_COPYMOVE(row);
        row()  = default;
        ~row() = default;

        std::string_view operator[](std::size_t index) const
        {
            return std::string_view(&m_line[m_data[index] + 1], m_data[index + 1] -  (m_data[index] + 1));
        }

        std::size_t size() const
        {
            return m_data.size() - 1;
        }

        void readNextRow(std::istream& str)
        {
            std::getline(str, m_line);

            m_data.clear();
            m_data.emplace_back(-1);
            std::string::size_type pos = 0;
            while((pos = m_line.find(',', pos)) != std::string::npos)
            {
                m_data.emplace_back(pos);
                ++pos;
            }
            // This checks for a trailing comma with no data after it.
            pos   = m_line.size();
            m_data.emplace_back(pos);
        }
    private:
        std::string      m_line;
        std::vector<int> m_data;
    };

    std::istream& operator>>(std::istream& str, row& data)
    {
        data.readNextRow(str);
        return str;
    }

    class iterator
    {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = row;
        using difference_type = std::size_t;
        using pointer = row*;
        using reference = row&;

        DEFAULT_COPYMOVE(iterator);
        iterator(std::istream& str):
            m_str(str.good() ? &str : nullptr)
        {
            ++(*this);
        }

        iterator():
            m_str(nullptr)
        {
        }

        ~iterator() = default;

        // Pre Increment
        iterator& operator++()
        {
            if (m_str)
            {
                if (!((*m_str) >> m_row))
                {
                    m_str = nullptr;
                }
            }
            return *this;
        }
        // Post increment
        iterator operator++(int)
        {
            iterator tmp(*this);
            ++(*this);
            return tmp;
        }

        row const& operator*() const
        {
            return m_row;
        }

        row const* operator->() const
        {
            return &m_row;
        }

        bool operator==(iterator const& rhs) const
        {
            return ((this == &rhs) || ((this->m_str == nullptr) && (rhs.m_str == nullptr)));
        }

        bool operator!=(iterator const& rhs) const
        {
            return !((*this) == rhs);
        }
    private:
        std::istream*    m_str;
        row              m_row;
    };

    class range
    {
        std::istream&   stream;
    public:
        range(std::istream& str) :
            stream(str)
        {}
        NO_COPYMOVE(range);
        ~range() = default;

        iterator begin() const
        {
            return iterator{stream};
        }

        iterator end() const
        {
            return iterator{};
        }
    };
}
