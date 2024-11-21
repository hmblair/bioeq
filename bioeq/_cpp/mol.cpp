#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>

namespace py = pybind11;
using namespace torch::indexing;
using std::vector;

torch::Tensor _singleMoleculeConnectedSubgraphs(
    torch::Tensor bonds,
    uint64_t numEdges
) {

    // Initialise the connected subgraphs with the bonds
    torch::Tensor tuples = bonds;
    for (uint64_t ix = 2; ix < numEdges; ix++) {
        // Overlap the last ix-1 indices with the first ix-1 indices,
        auto left_bonds = tuples.index(
            {Slice(), None, Slice(1, None, None)}
        );
        auto right_bonds = tuples.index(
            {None, Slice(), Slice(None, -1, None)}
        );
        // and check for any matches
        auto match_indices = at::nonzero(
            left_bonds == right_bonds
        );
        // Find the tuples that contain the overlaps
        auto tuple_matches = tuples.index({match_indices});
        // Merge the overlapping tuples to get a connected subgraph
        // with ix nodes
        tuples = torch::cat(
            {
            tuple_matches.index({Slice(), 0}),
            tuple_matches.index({Slice(), 1, Slice(-1)})
            }, -1
        );
    }

    return tuples;

}


torch::Tensor _connectedSubgraphs(
    torch::Tensor bonds,
    torch::Tensor moleculeSizes,
    uint64_t numEdges
) {

    torch::Tensor paddedMoleculeSizes = torch::cat(
        {torch::tensor({0}), moleculeSizes}, 0
    );
    int64_t low = 0, high = 0;
    std::vector<torch::Tensor> tuplesList;

    for (int64_t ix = 0; ix < moleculeSizes.size(0); ix++) {
        // Update the position of the new molecule in the bonds tensor
        low += paddedMoleculeSizes[ix].item<int64_t>();
        high += paddedMoleculeSizes[ix + 1].item<int64_t>();
        // Subset the bonds corresponding to this molecule
        torch::Tensor bondsSubset = bonds.index(
            {Slice(low, high)}
        );
        // Compute the connected subgraphs for this molecule
        tuplesList.push_back(
            _singleMoleculeConnectedSubgraphs(bondsSubset, numEdges)
        );

    }

    // Stack the subgraphs of each molecule
    return torch::cat(tuplesList, 0);
}

torch::Tensor _partitionCount(torch::Tensor partitions, torch::Tensor values) {
    // Ensure inputs are on CPU
    partitions = partitions.cpu();
    values = values.cpu();

    // Access raw pointers for efficient processing
    auto partitions_data = partitions.data_ptr<int64_t>();
    auto values_data = values.data_ptr<int64_t>();

    int64_t num_partitions = partitions.size(0);
    int64_t num_values = values.size(0);

    std::vector<int64_t> partition_counts(num_values);

    int64_t partitionIndex = 0;

    // Process each value in `values`
    for (int64_t i = 0; i < num_values; ++i) {
        int64_t value = values_data[i];
        int64_t count = 0;
        int64_t cumulative = 0;

        // Accumulate partitions until the cumulative sum meets or exceeds `value`
        while (partitionIndex < num_partitions && cumulative < value) {
            cumulative += partitions_data[partitionIndex];
            partitionIndex++;
            count++;
        }

        if (cumulative != value) {
            throw std::runtime_error(
                "Partitioning error: cumulative sum does not match value. "
                "Value at index " + std::to_string(i) + 
                " has cumulative sum " + std::to_string(cumulative) + 
                " but expected " + std::to_string(value) + "."
            );
        }

        partition_counts[i] = count;
    }

    // Convert result to a tensor
    return torch::tensor(partition_counts, torch::dtype(torch::kInt64).device(values.device()));
}

