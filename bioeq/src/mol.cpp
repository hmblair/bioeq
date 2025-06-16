#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>

using namespace torch::indexing;
using std::vector;


torch::Tensor _connectedSubgraphs(
    torch::Tensor bonds,
    uint64_t numNodes
) {

    // Initialise the connected subgraphs with the bonds
    torch::Tensor tuples = bonds;
    // If numNodes is less than 2, then throw an error
    if (numNodes < 2) {
        throw std::runtime_error(
            "Expected at least 2 nodes; got " + std::to_string(numNodes) + ".\n"
        );
    }
    // If numNodes is 2, then there is nothing to be done
    else if (numNodes == 2) {
        return tuples;
    }

    // Otherwise, we compute a mapping from an edge to its source nodes
    torch::Tensor src = tuples.index({Slice(), 0});
    torch::Tensor dst = tuples.index({Slice(), 1});
    int64_t totNodes = dst.max().item<int64_t>() + 1;
    int64_t totEdges = dst.size(0);
    // Get a vector containing the indices all copies of each node
    // in src
    vector<vector<int64_t>> nodeIX(totNodes);
    for (int64_t jx = 0; jx < totEdges; jx++) {
        nodeIX[src[jx].item<int64_t>()].push_back(jx);
    }
    torch::Tensor bc = torch::bincount(src);

    for (int64_t e_ix = 2; e_ix < numNodes; e_ix++) {

        // Get the final two nodes in all previous subgraphs
        torch::Tensor nodes = tuples.index({Slice(), -1});
        totEdges = nodes.size(0);
        // For each node in dst, we will create a set of indices
        // denoting the positions it appears in in src, so that we can
        // find all edges that it connects to.
        vector<int64_t> ix(
            bc.index({nodes}).sum().item<int64_t>()
        );
        // Loop over the sizes and fill in ix
        int64_t pos = 0;
        for (int64_t jx = 0; jx < totEdges; jx++) {
            for (int64_t s : nodeIX[nodes[jx].item<int64_t>()]) {
                ix[pos] = s;
                pos++;
            }
        }
        // We must expand the subgraphs according to how many nodes
        // each node in dst was connected to
        torch::Tensor prev_indices = bc.index({nodes});
        torch::Tensor indices = torch::tensor(ix);
        // Join the newly-discovered nodes with the previous subgraphs
        tuples = torch::cat(
            {tuples.repeat_interleave(prev_indices, 0), dst.index({indices}).unsqueeze(1)}, 1
        );
        // Ensure that all nodes are unique
        for (int64_t nd = 0; nd < e_ix; nd++) {
            tuples = tuples.index(
               {tuples.index({Slice(), nd}) != tuples.index({Slice(), -1})}
            );
        }
    }

    return tuples.index(
        {tuples.index({Slice(), 0}) < tuples.index({Slice(), -1})}
    );;

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
            (left_bonds == right_bonds).all(-1)
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
        // Remove any tuples with duplicate nodes. Since the left and
        // right matches are guaranteed to have unique values, we need
        // only compare the first and the last.
        tuples = tuples.index(
            {tuples.index({Slice(), 0}) != tuples.index({Slice(), -1})}
        );

    }

    return tuples;

}


torch::Tensor _connectedSubgraphsOld(
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
    torch::Tensor subgraphs = torch::cat(tuplesList, 0);
    // Ensure uniqueness by choosing one of the two possible orientations of each subgraph.
    return subgraphs.index(
        {subgraphs.index({Slice(), 0}) < subgraphs.index({Slice(), -1})}
    );
}

std::vector<torch::Tensor> _pairwiseAtomSum(
    torch::Tensor val1,
    torch::Tensor val2,
    torch::Tensor val3,
    torch::Tensor pairwiseTypes
) {

    int64_t numTypes = pairwiseTypes.max().item<int64_t>() + 1;
    torch::Tensor singleSum = torch::zeros(
        {numTypes}
    );
    torch::Tensor pairwiseSum = torch::zeros(
        {numTypes, numTypes}
    );

    float* singleSum_ptr = singleSum.data_ptr<float>();
    float* pairwiseSum_ptr = pairwiseSum.data_ptr<float>();
    int64_t* pairwiseTypes_ptr = pairwiseTypes.data_ptr<int64_t>();
    float* val1_ptr = val1.data_ptr<float>();
    float* val2_ptr = val2.data_ptr<float>();
    float* val3_ptr = val3.data_ptr<float>();
    
    int64_t numAtoms = val1.size(0);
    for (int64_t ix = 0; ix < numAtoms; ix++) {
        std::cout << ix << "\n";
        for (int64_t jx = 0; jx < numAtoms; jx++) {

            singleSum_ptr[
                pairwiseTypes_ptr[ix * numAtoms + jx]
            ] += val3_ptr[ix * numAtoms + jx];

            for (int64_t kx = 0; kx < numAtoms; kx++) {
                pairwiseSum_ptr[
                    pairwiseTypes_ptr[ix * numAtoms + jx] * numTypes + pairwiseTypes_ptr[ix * numAtoms + kx]
                ] += val1_ptr[ix * numAtoms + jx] * val2_ptr[ix * numAtoms + kx];
            }
        }
    }

    return {singleSum, pairwiseSum};

}

std::vector<torch::Tensor> _GetSQ(
    torch::Tensor val_S,
    torch::Tensor val_Q,
    torch::Tensor pairwiseTypes,
    torch::Tensor bonds
) {

    int64_t numTypes = pairwiseTypes.max().item<int64_t>() + 1;
    int64_t numAtoms = bonds.max().item<int64_t>() + 1;

    torch::Tensor S = torch::zeros({numTypes});
    torch::Tensor Q = torch::zeros({numTypes, 3 * numAtoms});

    int64_t* pairwiseTypes_ptr = pairwiseTypes.data_ptr<int64_t>();
    float* val_S_ptr = val_S.data_ptr<float>();
    float* val_Q_ptr = val_Q.data_ptr<float>();
    float* S_ptr = S.data_ptr<float>();
    float* Q_ptr = Q.data_ptr<float>();
    int64_t* bond_ptr = bonds.data_ptr<int64_t>();

    int64_t _n_pts = 3 * numAtoms;
    int64_t numBonds = bonds.size(0);
    for (int64_t ix = 0; ix < numBonds; ix++) {

        int64_t u = bond_ptr[2 * ix];
        int64_t v = bond_ptr[2 * ix + 1];
        if (u == v) { continue; }

        int64_t ptype = pairwiseTypes_ptr[ix];
        S_ptr[ptype] += val_S_ptr[ix];

        for (int64_t kx = 0; kx < 3; kx++){
            Q_ptr[ptype * _n_pts + 3 * u + kx] += val_Q_ptr[3 * ix + kx];
        }

    }

    return {S, Q};

}


std::vector<torch::Tensor> _GetSQH(
    torch::Tensor val_S,
    torch::Tensor val_Q,
    torch::Tensor val_H,
    torch::Tensor pairwiseTypes,
    torch::Tensor bonds
) {

    int64_t numTypes = pairwiseTypes.max().item<int64_t>() + 1;
    int64_t numAtoms = bonds.max().item<int64_t>() + 1;

    torch::Tensor S = torch::zeros(
        {numTypes}
    );
    torch::Tensor Q = torch::zeros(
        {numTypes, 3 * numAtoms}
    );
    torch::Tensor H = torch::zeros(
        {numTypes, 3 * numAtoms, 3 * numAtoms}
    );

    int64_t* pairwiseTypes_ptr = pairwiseTypes.data_ptr<int64_t>();
    float* val_S_ptr = val_S.data_ptr<float>();
    float* val_Q_ptr = val_Q.data_ptr<float>();
    float* val_H_ptr = val_H.data_ptr<float>();
    float* S_ptr = S.data_ptr<float>();
    float* Q_ptr = Q.data_ptr<float>();
    float* H_ptr = H.data_ptr<float>();
    int64_t* bond_ptr = bonds.data_ptr<int64_t>();

    int64_t _n_pts = 3 * numAtoms;

    int64_t numBonds = bonds.size(0);
    for (int64_t ix = 0; ix < numBonds; ix++) {

        int64_t u = bond_ptr[2 * ix];
        int64_t v = bond_ptr[2 * ix + 1];
        if (u == v) { continue; }

        int64_t ptype = pairwiseTypes_ptr[ix];

        S_ptr[ptype] += val_S_ptr[ix];

        for (int64_t kx = 0; kx < 3; kx++) {

            Q_ptr[ptype * _n_pts + 3 * u + kx] += val_Q_ptr[3 * ix + kx];

            for (int64_t lx = 0; lx < 3; lx++) {

                int64_t offset = (u * 3 + kx) * 3 * numAtoms + (v * 3 + lx);
                int64_t offset_diag = (u * 3 + kx) * 3 * numAtoms + (u * 3 + lx);
                int64_t H_offset = ix * 9 + kx * 3 + lx;

                H_ptr[ptype * _n_pts * _n_pts + offset] -= val_H_ptr[H_offset];
                H_ptr[ptype * _n_pts * _n_pts + offset_diag] += val_H_ptr[H_offset];

            }

        }

    }

    return {S, Q, H};

}

