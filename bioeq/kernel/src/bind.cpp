
#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

using namespace at;
using std::vector;

torch::Tensor matmulStepOneCUDA(
    torch::Tensor basis,
    torch::Tensor features,
    torch::Tensor src,
    torch::Tensor in_blocks
);

torch::Tensor matmulStepTwoCUDA(
    torch::Tensor inter,
    torch::Tensor rw,
    vector<int64_t> degrees
);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor matmulStepOne(
    torch::Tensor basis,
    torch::Tensor features,
    torch::Tensor src,
    torch::Tensor in_blocks
    ) {

        // check input
        // CHECK_INPUT(basis);
        // CHECK_INPUT(features);
        // CHECK_INPUT(src);
        // CHECK_INPUT(out_blocks);
        // CHECK_INPUT(in_blocks);

        return matmulStepOneCUDA(
            basis,
            features,
            src,
            in_blocks
        );
    }


torch::Tensor matmulStepTwo(
    torch::Tensor inter,
    torch::Tensor rw,
    vector<int64_t> degrees
    ) {

        // check input
        // CHECK_INPUT(inter);
        // CHECK_INPUT(rw);

        return matmulStepTwoCUDA(
            inter,
            rw,
            degrees
        );
    }


//
// pybind11 binding
//


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmulStepOne", &matmulStepOne, "matmulStepOne");
    m.def("matmulStepTwo", &matmulStepTwo, "matmulStepTwo");
}
