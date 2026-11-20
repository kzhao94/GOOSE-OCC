#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// ===== 辅助函数 =====

// 原子加 float3
__device__ void atomicAdd(float3* address, float3 val) {
    atomicAdd(&(address->x), val.x);
    atomicAdd(&(address->y), val.y);
    atomicAdd(&(address->z), val.z);
}

// 计算马氏距离（向量化版本）
__device__ float mahalanobis_distance(
    const float* diff, const float* prec
) {
    float x = diff[0], y = diff[1], z = diff[2];
    float dx = prec[0]*x + prec[1]*y + prec[2]*z;
    float dy = prec[3]*x + prec[4]*y + prec[5]*z;
    float dz = prec[6]*x + prec[7]*y + prec[8]*z;
    return dx*x + dy*y + dz*z;
}

// 快速边界检查
__device__ __forceinline__ bool in_bounds(int x, int y, int z, int W, int H, int D) {
    return (unsigned)x < (unsigned)W &&
           (unsigned)y < (unsigned)H &&
           (unsigned)z < (unsigned)D;
}

// ===== 方法1：动态Bounds Kernel =====
// 【特点】
// - 精度：★★★★★ (理论完美，精确计算3σ边界)
// - 速度：★★☆☆☆ (浮点边界计算开销大)
// - 适用：高精度场景、验证基准
// 【原理】根据每个高斯的协方差矩阵，动态计算其世界坐标系下的3σ包围盒，
//        然后映射到体素空间进行遍历。保证不遗漏任何有效体素。

__global__ void dynamic_kernel(
    const float* __restrict__ means3d,
    const float* __restrict__ opacities,
    const float* __restrict__ precisions,
    const float* __restrict__ features,
    const float* __restrict__ covariances,
    const float* __restrict__ vol_range,
    float voxel_size,
    int N, int C, int W, int H, int D,
    float* __restrict__ density_out,
    float* __restrict__ feats_out
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    // 加载数据（使用 __ldg 利用只读缓存）
    float mean_x = __ldg(&means3d[gid * 3 + 0]);
    float mean_y = __ldg(&means3d[gid * 3 + 1]);
    float mean_z = __ldg(&means3d[gid * 3 + 2]);
    float opacity = __ldg(&opacities[gid]);

    const float* prec = &precisions[gid * 9];
    const float* cov = &covariances[gid * 9];

    // 计算世界坐标系下的3σ边界
    float sigma_x = sqrtf(cov[0]);
    float sigma_y = sqrtf(cov[4]);
    float sigma_z = sqrtf(cov[8]);

    float bound_x_min = fmaxf(mean_x - 3.0f * sigma_x, vol_range[0]);
    float bound_y_min = fmaxf(mean_y - 3.0f * sigma_y, vol_range[1]);
    float bound_z_min = fmaxf(mean_z - 3.0f * sigma_z, vol_range[2]);
    float bound_x_max = fminf(mean_x + 3.0f * sigma_x, vol_range[3]);
    float bound_y_max = fminf(mean_y + 3.0f * sigma_y, vol_range[4]);
    float bound_z_max = fminf(mean_z + 3.0f * sigma_z, vol_range[5]);

    // 转换为体素索引
    int lo_x = max(0, (int)((bound_x_min - vol_range[0]) / voxel_size));
    int lo_y = max(0, (int)((bound_y_min - vol_range[1]) / voxel_size));
    int lo_z = max(0, (int)((bound_z_min - vol_range[2]) / voxel_size));
    int hi_x = min(W, (int)((bound_x_max - vol_range[0]) / voxel_size) + 1);
    int hi_y = min(H, (int)((bound_y_max - vol_range[1]) / voxel_size) + 1);
    int hi_z = min(D, (int)((bound_z_max - vol_range[2]) / voxel_size) + 1);

    // 遍历包围盒
    for (int x = lo_x; x < hi_x; x++) {
        for (int y = lo_y; y < hi_y; y++) {
            for (int z = lo_z; z < hi_z; z++) {
                float vox_x = vol_range[0] + (x + 0.5f) * voxel_size;
                float vox_y = vol_range[1] + (y + 0.5f) * voxel_size;
                float vox_z = vol_range[2] + (z + 0.5f) * voxel_size;

                float dx = vox_x - mean_x;
                float dy = vox_y - mean_y;
                float dz = vox_z - mean_z;

                float maha = prec[0]*dx*dx + prec[4]*dy*dy + prec[8]*dz*dz
                           + 2*prec[1]*dx*dy + 2*prec[2]*dx*dz + 2*prec[5]*dy*dz;
                float density = opacity * expf(-0.5f * maha);

                if (density < 1e-6f) continue;

                int idx = x * H * D + y * D + z;
                atomicAdd(&density_out[idx], density);

                if (feats_out != nullptr) {
                    for (int c = 0; c < C; c++) {
                        float feat_val = __ldg(&features[gid * C + c]) * density;
                        atomicAdd(&feats_out[idx * C + c], feat_val);
                    }
                }
            }
        }
    }
}

// ===== 方法2：固定邻域 Kernel =====
// 【特点】
// - 精度：★★★☆☆ (对大高斯可能有截断误差)
// - 速度：★★★★☆ (循环固定，性能稳定)
// - 适用：实时应用、高斯尺度受控的场景
// 【原理】对所有高斯使用固定半径的邻域遍历，不考虑高斯实际大小。
//        实现简单高效，但可能对小高斯浪费计算，对大高斯产生截断。

__global__ void fixed_kernel(
    const float* __restrict__ means3d,
    const float* __restrict__ opacities,
    const float* __restrict__ precisions,
    const float* __restrict__ features,
    const float* __restrict__ vol_min,
    float voxel_size,
    int N, int C, int W, int H, int D,
    int max_neighborhood,
    float* __restrict__ density_out,
    float* __restrict__ feats_out
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    float mean_x = __ldg(&means3d[gid * 3 + 0]);
    float mean_y = __ldg(&means3d[gid * 3 + 1]);
    float mean_z = __ldg(&means3d[gid * 3 + 2]);
    float opacity = __ldg(&opacities[gid]);
    const float* prec = &precisions[gid * 9];

    int base_x = (int)((mean_x - vol_min[0]) / voxel_size);
    int base_y = (int)((mean_y - vol_min[1]) / voxel_size);
    int base_z = (int)((mean_z - vol_min[2]) / voxel_size);

    // 固定邻域循环
    for (int dx = -max_neighborhood; dx <= max_neighborhood; dx++) {
        int x = base_x + dx;
        if ((unsigned)x >= (unsigned)W) continue;

        for (int dy = -max_neighborhood; dy <= max_neighborhood; dy++) {
            int y = base_y + dy;
            if ((unsigned)y >= (unsigned)H) continue;

            for (int dz = -max_neighborhood; dz <= max_neighborhood; dz++) {
                int z = base_z + dz;
                if ((unsigned)z >= (unsigned)D) continue;

                float vox_x = vol_min[0] + (x + 0.5f) * voxel_size;
                float vox_y = vol_min[1] + (y + 0.5f) * voxel_size;
                float vox_z = vol_min[2] + (z + 0.5f) * voxel_size;

                float dx_ = vox_x - mean_x;
                float dy_ = vox_y - mean_y;
                float dz_ = vox_z - mean_z;

                float p_dx = prec[0]*dx_ + prec[1]*dy_ + prec[2]*dz_;
                float p_dy = prec[3]*dx_ + prec[4]*dy_ + prec[5]*dz_;
                float p_dz = prec[6]*dx_ + prec[7]*dy_ + prec[8]*dz_;
                float maha = p_dx*dx_ + p_dy*dy_ + p_dz*dz_;

                float density = opacity * expf(-0.5f * maha);
                if (density < 1e-6f) continue;

                int idx = x * H * D + y * D + z;
                atomicAdd(&density_out[idx], density);

                if (feats_out != nullptr) {
                    for (int c = 0; c < C; c++) {
                        float feat_val = __ldg(&features[gid * C + c]) * density;
                        atomicAdd(&feats_out[idx * C + c], feat_val);
                    }
                }
            }
        }
    }
}

// ===== 方法3：自适应邻域 Kernel =====
// 【特点】
// - 精度：★★★★☆ (在max_neighborhood限制内达到最优)
// - 速度：★★★★☆ (小高斯加速明显，大高斯性能稳定)
// - 适用：推荐使用！平衡精度与效率的最佳方案
// 【原理】根据高斯实际尺寸动态计算循环半径，小高斯少循环，大高斯多循环，
//        但设置max_neighborhood上限防止极端情况。兼具精度与效率。

__global__ void adaptive_kernel(
    const float* __restrict__ means3d,
    const float* __restrict__ opacities,
    const float* __restrict__ precisions,
    const float* __restrict__ features,
    const float* __restrict__ covariances,
    const float* __restrict__ vol_min,
    float voxel_size,
    int N, int C, int W, int H, int D,
    int max_neighborhood,
    float* __restrict__ density_out,
    float* __restrict__ feats_out
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    // 加载数据
    float mean_x = __ldg(&means3d[gid * 3 + 0]);
    float mean_y = __ldg(&means3d[gid * 3 + 1]);
    float mean_z = __ldg(&means3d[gid * 3 + 2]);
    float opacity = __ldg(&opacities[gid]);
    const float* prec = &precisions[gid * 9];
    const float* cov = &covariances[gid * 9];

    // 计算自适应半径
    float sigma_x = sqrtf(cov[0]);
    float sigma_y = sqrtf(cov[4]);
    float sigma_z = sqrtf(cov[8]);

    int radius_x = min((int)ceilf(3.0f * sigma_x / voxel_size), max_neighborhood);
    int radius_y = min((int)ceilf(3.0f * sigma_y / voxel_size), max_neighborhood);
    int radius_z = min((int)ceilf(3.0f * sigma_z / voxel_size), max_neighborhood);

    int base_x = (int)((mean_x - vol_min[0]) / voxel_size);
    int base_y = (int)((mean_y - vol_min[1]) / voxel_size);
    int base_z = (int)((mean_z - vol_min[2]) / voxel_size);

    // 自适应邻域循环
    for (int dx = -radius_x; dx <= radius_x; dx++) {
        int x = base_x + dx;
        if ((unsigned)x >= (unsigned)W) continue;

        for (int dy = -radius_y; dy <= radius_y; dy++) {
            int y = base_y + dy;
            if ((unsigned)y >= (unsigned)H) continue;

            for (int dz = -radius_z; dz <= radius_z; dz++) {
                int z = base_z + dz;
                if ((unsigned)z >= (unsigned)D) continue;

                float vox_x = vol_min[0] + (x + 0.5f) * voxel_size;
                float vox_y = vol_min[1] + (y + 0.5f) * voxel_size;
                float vox_z = vol_min[2] + (z + 0.5f) * voxel_size;

                float dx_ = vox_x - mean_x;
                float dy_ = vox_y - mean_y;
                float dz_ = vox_z - mean_z;

                float p_dx = prec[0]*dx_ + prec[1]*dy_ + prec[2]*dz_;
                float p_dy = prec[3]*dx_ + prec[4]*dy_ + prec[5]*dz_;
                float p_dz = prec[6]*dx_ + prec[7]*dy_ + prec[8]*dz_;
                float maha = p_dx*dx_ + p_dy*dy_ + p_dz*dz_;

                float density = opacity * expf(-0.5f * maha);
                if (density < 1e-6f) continue;

                int idx = x * H * D + y * D + z;
                atomicAdd(&density_out[idx], density);

                if (feats_out != nullptr) {
                    for (int c = 0; c < C; c++) {
                        float feat_val = __ldg(&features[gid * C + c]) * density;
                        atomicAdd(&feats_out[idx * C + c], feat_val);
                    }
                }
            }
        }
    }
}

// ===== Host Functions =====

std::vector<torch::Tensor> dynamic_cuda_forward(
    torch::Tensor means3d,
    torch::Tensor opacities,
    torch::Tensor precisions,
    torch::Tensor features,
    torch::Tensor covariances,
    torch::Tensor vol_range,
    float voxel_size,
    std::vector<int64_t> grid_shape
) {
    int N = means3d.size(0);
    int C = features.size(1);
    int W = grid_shape[0], H = grid_shape[1], D = grid_shape[2];

    auto density_out = torch::zeros({W, H, D, 1},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto feats_out = torch::zeros({W, H, D, C},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    dynamic_kernel<<<blocks, threads>>>(
        means3d.data_ptr<float>(),
        opacities.data_ptr<float>(),
        precisions.data_ptr<float>(),
        features.data_ptr<float>(),
        covariances.data_ptr<float>(),
        vol_range.data_ptr<float>(),
        voxel_size, N, C, W, H, D,
        density_out.data_ptr<float>(),
        feats_out.data_ptr<float>()
    );

    cudaDeviceSynchronize();
    return {density_out, feats_out};
}

std::vector<torch::Tensor> fixed_cuda_forward(
    torch::Tensor means3d,
    torch::Tensor opacities,
    torch::Tensor precisions,
    torch::Tensor features,
    torch::Tensor vol_range,
    float voxel_size,
    std::vector<int64_t> grid_shape,
    int max_neighborhood
) {
    int N = means3d.size(0);
    int C = features.size(1);
    int W = grid_shape[0], H = grid_shape[1], D = grid_shape[2];

    auto density_out = torch::zeros({W, H, D, 1},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto feats_out = torch::zeros({W, H, D, C},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    fixed_kernel<<<blocks, threads>>>(
        means3d.data_ptr<float>(),
        opacities.data_ptr<float>(),
        precisions.data_ptr<float>(),
        features.data_ptr<float>(),
        vol_range.data_ptr<float>(),
        voxel_size, N, C, W, H, D, max_neighborhood,
        density_out.data_ptr<float>(),
        feats_out.data_ptr<float>()
    );

    cudaDeviceSynchronize();
    return {density_out, feats_out};
}

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
) {
    int N = means3d.size(0);
    int C = features.size(1);
    int W = grid_shape[0], H = grid_shape[1], D = grid_shape[2];

    auto density_out = torch::zeros({W, H, D, 1},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto feats_out = torch::zeros({W, H, D, C},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    adaptive_kernel<<<blocks, threads>>>(
        means3d.data_ptr<float>(),
        opacities.data_ptr<float>(),
        precisions.data_ptr<float>(),
        features.data_ptr<float>(),
        covariances.data_ptr<float>(),
        vol_range.data_ptr<float>(),
        voxel_size, N, C, W, H, D, max_neighborhood,
        density_out.data_ptr<float>(),
        feats_out.data_ptr<float>()
    );

    cudaDeviceSynchronize();
    return {density_out, feats_out};
}