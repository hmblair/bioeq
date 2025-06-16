#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>

namespace py = pybind11;
using namespace torch::indexing;

torch::Tensor _connectedSubgraphsOld(
    torch::Tensor bonds,
    torch::Tensor moleculeSizes,
    uint64_t numEdges
);

torch::Tensor _connectedSubgraphs(
    torch::Tensor bonds,
    uint64_t numEdges
);

torch::Tensor _partitionCount(
    torch::Tensor partitions,
    torch::Tensor values
);

std::vector<torch::Tensor> _pairwiseAtomSum(
    torch::Tensor val1,
    torch::Tensor val2,
    torch::Tensor val3,
    torch::Tensor pairwiseTypes
);

std::vector<torch::Tensor> _GetSQ(
    torch::Tensor val_S,
    torch::Tensor val_Q,
    torch::Tensor pairwiseTypes,
    torch::Tensor bonds
);

std::vector<torch::Tensor> _GetSQH(
    torch::Tensor val_S,
    torch::Tensor val_Q,
    torch::Tensor val_H,
    torch::Tensor pairwiseTypes,
    torch::Tensor bonds
);

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_connectedSubgraphs", &_connectedSubgraphs, "None");
    m.def("_connectedSubgraphsOld", &_connectedSubgraphsOld, "None");
    m.def("_partitionCount", &_partitionCount, "None");
    m.def("_pairwiseAtomSum", &_pairwiseAtomSum, "None");
    m.def("_GetSQ", &_GetSQ, "None");
    m.def("_GetSQH", &_GetSQH, "None");
}

