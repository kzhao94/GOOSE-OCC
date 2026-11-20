#include <torch/extension.h>
#include <vector>

// 声明 CUDA kernel
std::vector<torch::Tensor> dynamic_cuda_forward(
    torch::Tensor means3d,
    torch::Tensor opacities,
    torch::Tensor precisions,
    torch::Tensor features,
    torch::Tensor covariances,
    torch::Tensor vol_range,
    float voxel_size,
    std::vector<int64_t> grid_shape
);

std::vector<torch::Tensor> fixed_cuda_forward(
    torch::Tensor means3d,
    torch::Tensor opacities,
    torch::Tensor precisions,
    torch::Tensor features,
    torch::Tensor vol_range,
    float voxel_size,
    std::vector<int64_t> grid_shape,
    int max_neighborhood
);

std::vector<torch::Tensor> adaptive_cuda_forward(
    torch::Tensor means3d,
    torch::Tensor opacities,
    torch::Tensor precisions,
    torch::Tensor features,
    torch::Tensor covariances,
    torch::Tensor vol_range,
    float voxel_size,
    std::vector<int64_t> grid_shape,
    int max_neighborhood
);

// Python 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_dynamic", &dynamic_cuda_forward,
          "Dynamic Bounds Voxel Splat (High Precision)");
    m.def("forward_fixed", &fixed_cuda_forward,
          "Fixed Neighborhood Voxel Splat (High Speed)");
    m.def("forward_adaptive", &adaptive_cuda_forward,
          "Adaptive Neighborhood Voxel Splat (Balanced)");
}
