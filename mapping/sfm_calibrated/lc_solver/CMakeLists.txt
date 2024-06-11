cmake_minimum_required(VERSION 3.0)
project(lc_solver)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fbow REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${fbow_INCLUDE_DIRS})

pybind11_add_module(lc_solver fbow.cpp)
target_link_libraries(lc_solver PRIVATE ${OpenCV_LIBRARIES} ${fbow_LIBS})
target_link_libraries(lc_solver PRIVATE pybind11::module)

install(TARGETS lc_solver LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
