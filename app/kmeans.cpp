#include <torch/torch.h>
#include <torch/script.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>

namespace py = pybind11;

class KMeansModule : public torch::nn::Module {
private:
    torch::Tensor centroids;
    int k;
    int max_iters;

public:
    KMeansModule(int k, int max_iters) : k(k), max_iters(max_iters) {}

    void fit(torch::Tensor data) {
        auto options = torch::TensorOptions().dtype(data.dtype()).device(data.device());
        centroids = data.index({torch::randperm(data.size(0)).slice(0, 0, k)}).clone();
        
        for (int i = 0; i < max_iters; ++i) {
            auto distances = torch::cdist(data, centroids);
            auto labels = std::get<1>(distances.min(1));
            
            for (int j = 0; j < k; ++j) {
                auto mask = (labels == j);
                if (mask.sum().item<int>() > 0) {
                    centroids[j] = data.index({mask}).mean(0);
                }
            }
        }
    }

    torch::Tensor predict(torch::Tensor data) {
        auto distances = torch::cdist(data, centroids);
        return std::get<1>(distances.min(1));
    }

    torch::Tensor get_centroids() {
        return centroids;
    }
};

PYBIND11_MODULE(kmeans, m) {
    py::class_<KMeansModule, std::shared_ptr<KMeansModule>>(m, "KMeans")
        .def(py::init<int, int>())
        .def("fit", &KMeansModule::fit)
        .def("predict", &KMeansModule::predict)
        .def("get_centroids", &KMeansModule::get_centroids);
}
