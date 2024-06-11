#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "fbow/fbow.h"

namespace py = pybind11;

int test() {
    return 123;
}

// Helper function to convert NumPy array to cv::Mat
cv::Mat numpy_array_to_cv_mat(const py::array_t<unsigned char>& array) {
    py::buffer_info buf_info = array.request();
    int rows = buf_info.shape[0];
    int cols = buf_info.shape[1];
    cv::Mat mat(rows, cols, CV_8UC(1), buf_info.ptr);
    return mat;
}

// Binding for fBow structure
void bind_fBow(py::module &m) {
    py::class_<fbow::fBow>(m, "fBow")
        .def(py::init<>())
        .def("toStream", &fbow::fBow::toStream)
        .def("fromStream", &fbow::fBow::fromStream)
        .def("hash", &fbow::fBow::hash)
        .def_static("score", &fbow::fBow::score);
}

// Binding for Vocabulary class
void bind_Vocabulary(py::module &m) {
    py::class_<fbow::Vocabulary>(m, "Vocabulary")
        .def(py::init<>())
        .def("readFromFile", &fbow::Vocabulary::readFromFile)
        .def("saveToFile", &fbow::Vocabulary::saveToFile)
        .def("getDescType", &fbow::Vocabulary::getDescType)
        .def("getDescSize", &fbow::Vocabulary::getDescSize)
        .def("getDescName", &fbow::Vocabulary::getDescName)
        .def("getK", &fbow::Vocabulary::getK)
        .def("isValid", &fbow::Vocabulary::isValid)
        .def("size", &fbow::Vocabulary::size)
        .def("clear", &fbow::Vocabulary::clear)
        .def("hash", &fbow::Vocabulary::hash)
        .def("transform", 
            [](fbow::Vocabulary &vocab, const py::array_t<unsigned char>& array) {
                cv::Mat features = numpy_array_to_cv_mat(array);
                return vocab.transform(features);
            }, py::arg("features"))
        .def("transform", 
            [](fbow::Vocabulary &vocab, const py::array_t<unsigned char>& array, int level) {
                cv::Mat features = numpy_array_to_cv_mat(array);
                fbow::fBow result;
                fbow::fBow2 result2;
                vocab.transform(features, level, result, result2);
                return std::make_tuple(result, result2);
            }, py::arg("features"), py::arg("level"));
}

PYBIND11_MODULE(lc_solver, m) {
    m.def("test", &test, "");

    bind_fBow(m);
    bind_Vocabulary(m);
}
