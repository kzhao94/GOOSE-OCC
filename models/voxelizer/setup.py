import os
import warnings
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ===========================================
# 1. 定义编译参数
# ===========================================

nvcc_flags = [
    '-O3',
    '-use_fast_math',  # 加速数学运算
]

cxx_flags = [
    '-O3',
    '-std=c++17'
]

# 【关键修改】不再硬编码默认值，而是直接依赖 PyTorch 的自动检测
# 如果你需要强制指定，请设置环境变量 TORCH_CUDA_ARCH_LIST
# 例如: export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"
torch_cuda_arch_list = os.getenv("TORCH_CUDA_ARCH_LIST")

if torch_cuda_arch_list:
    print(f"User specified CUDA architectures: {torch_cuda_arch_list}")
    arch_list = torch_cuda_arch_list.split(';')
    for arch in arch_list:
        arch = arch.strip()
        if arch:
            arch_int = arch.replace('.', '')
            nvcc_flags.append(f'-gencode=arch=compute_{arch_int},code=sm_{arch_int}')
else:
    # 如果没有设置环境变量，打印提示，交给 PyTorch 的 BuildExtension 自动处理
    print("TORCH_CUDA_ARCH_LIST not set. Relying on PyTorch default architecture detection.")

# ===========================================
# 2. 设置扩展模块
# ===========================================

# 修改了包名：gaussian_voxelizer_cuda -> g2v
ext_modules = [
    CUDAExtension(
        name='g2v._C',  # 对应 import 路径: from g2v import _C
        sources=[
            'g2v/src/voxelizer_cuda.cpp',
            'g2v/src/voxelizer_cuda_kernel.cu',
        ],
        extra_compile_args={
            'cxx': cxx_flags,
            'nvcc': nvcc_flags,
        },
        include_dirs=[
            # 如果有自定义的头文件目录可以加在这里
            # os.path.abspath('g2v/include'),
        ]
    )
]

# ===========================================
# 3. Setup 主函数
# ===========================================

setup(
    name='g2v',  # pip install 后的名字
    version="1.0.0",
    description='High-performance Gaussian Voxelization CUDA Extension',
    author='Your Name',
    packages=find_packages(), # 会自动查找 g2v 目录
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    python_requires=">=3.7",
    install_requires=["torch"],
    zip_safe=False, # C 扩展通常不 zip
)
