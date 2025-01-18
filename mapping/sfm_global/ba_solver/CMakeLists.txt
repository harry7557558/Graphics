cmake_minimum_required(VERSION 3.0)
project(ba_solver)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(pybind11 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} ${CERES_INCLUDE_DIR})

pybind11_add_module(ba_solver ba_solver.cpp)
target_link_libraries(ba_solver PRIVATE ${CERES_LIBRARIES})
target_link_libraries(ba_solver PRIVATE pybind11::module)

install(TARGETS ba_solver LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
