cmake_minimum_required(VERSION 3.9)

project(learning_nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(learning_nn
    simple_nn.h
    rnd_nn.h
    mnist_loader.h
    main.cpp)


target_include_directories(learning_nn PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../my_includes)
target_include_directories(learning_nn PUBLIC ${CMAKE_CURRENT_LIST_DIR}/my_includes)
#find_package (OpenMP REQUIRED)
#target_include_directories(learning_nn PUBLIC ${OpenMP_CXX_INCLUDE_DIRS})


#find_package (Eigen3 3 REQUIRED NO_MODULE)
#target_compile_definitions(learning_nn PUBLIC EIGEN_HAS_CXX11)


find_package(cereal REQUIRED)
target_include_directories(learning_nn PUBLIC ${Cereal_CXX_INCLUDE_DIRS})

#target_compile_options(learning_nn PUBLIC ${OpenMP_CXX_FLAGS})
target_compile_options(learning_nn PUBLIC -g -O3 -march=native -frtti -fexceptions
                                          -Wpedantic -Wall -Wextra -Werror=return-type)

#Sanitizing and protection options
target_compile_options(learning_nn PUBLIC -fstack-protector-all)

target_compile_options(learning_nn PUBLIC -fsanitize=bounds -fsanitize=undefined)
target_link_libraries(learning_nn ubsan)

#target_compile_options(learning_nn PUBLIC -fsanitize=thread)
#target_link_libraries(learning_nn tsan)

#asan - address
#tsan - thread
#hwsan - hwaddres
#ubsan
#lsan


target_link_libraries(learning_nn tbb)
#target_link_libraries(learning_nn Eigen3::Eigen)
#target_link_libraries(learning_nn ${OpenMP_CXX_LIBRARIES})

