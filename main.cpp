#include "simple_nn.h"
#include <iostream>
#include <thread>
#include "mnist_loader.h"

int main()
{
    SimpleLayeredNN<mnist_loader::samples_t, mnist_loader::inputs_size,
                    20 * mnist_loader::outputs_size, 20 * mnist_loader::outputs_size, mnist_loader::outputs_size> nn;
    nn.random_weights();

    mnist_loader srcf("/home/alex/Work/learning_nn/mnist_dataset/mnist_train_100.csv");
    const auto& src = srcf.train_data();
    for (int epoche =0; epoche < 5; ++epoche)
        for (const auto& ex : src)
        {
            nn.train(0.3f, ex.first, ex.second);
        }

    mnist_loader test("/home/alex/Work/learning_nn/mnist_dataset/mnist_test_10.csv");
    for (const auto& t : test.train_data())
    {
        const auto r = nn.query(t.first);
        std::cout << "Expected:   " << mnist_loader::parse_output(t.second) << std::endl;
        std::cout << "Recognized: " << mnist_loader::parse_output(r) << std::endl << std::endl;
    }

    return 0;
}
