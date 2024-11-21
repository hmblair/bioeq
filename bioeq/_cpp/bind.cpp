#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>

namespace py = pybind11;
using namespace torch::indexing;

torch::Tensor _connectedSubgraphs(
    torch::Tensor bonds,
    torch::Tensor moleculeSizes,
    uint64_t numEdges
);

torch::Tensor _partitionCount(
    torch::Tensor partitions,
    torch::Tensor values
);

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_connectedSubgraphs", &_connectedSubgraphs, "None");
    m.def("_partitionCount", &_partitionCount, "None");
}

