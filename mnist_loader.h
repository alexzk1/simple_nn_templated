#pragma once

#include <fstream>
#include <string>
#include <charconv>
#include "csv_reader.h"
#include "matrix2d.h"

class mnist_loader
{
public:
    constexpr static size_t inputs_size  = 784;
    constexpr static size_t outputs_size = 10;
    using samples_t = float;
    using train_value = std::pair<VectorRow<samples_t, inputs_size>, VectorRow<samples_t, outputs_size>>;
private:
    std::vector<train_value> wholeData;
public:
    mnist_loader(const std::string& file_name)
    {
        std::ifstream fs(file_name);
        wholeData.reserve(100);
        for (const auto& example : csv::range(fs))
        {
            const auto get_int = [&example](int index)
            {
                const auto sv = example[index];
                int v = -1;
                std::from_chars(sv.data(), sv.data() + sv.size(), v);
                return v;
            };
            const auto sz = example.size();
            train_value val;
            val.second = make_output_vector(get_int(0));

            for (size_t i = 1; i < sz; ++i)
                *(val.first.begin() + i - 1) = (get_int(i) / static_cast<samples_t>(255))
                                               * static_cast<samples_t>(0.99) + static_cast<samples_t>(0.01);
            wholeData.push_back(std::move(val));
        }
    }
    ~mnist_loader() = default;

    const auto& train_data() const
    {
        return wholeData;
    }

    static VectorRow<samples_t, outputs_size> make_output_vector(int active)
    {
        VectorRow<samples_t, outputs_size> r;
        std::fill(std::begin(r), std::end(r), static_cast<samples_t>(0.001));
        r.at(active, 0) = static_cast<samples_t>(0.999);
        return r;
    }

    static std::string parse_output(const VectorRow<samples_t, outputs_size>& inp)
    {
        const auto it = std::max_element(inp.begin(), inp.end());
        const int d = std::distance(inp.begin(), it);
        return "Value: " + std::to_string(d) + "; with float = " + std::to_string(*it);
    }
};
