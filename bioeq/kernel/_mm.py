from __future__ import annotations
import torch
import torch.nn as nn
import dgl
from bioeq.geom import ProductRepr
from torch import autograd
from bioeq.kernel._C import (
    matmulStepOne,
    matmulStepTwo,
)


class _EquivariantMatmulKernel(autograd.Function):
    """
    Implements the matmul step in an equivariant transformer with a custom
    kernel backend for the forward and backward passes.
    """

    @staticmethod
    def forward(
        ctx,
        graph: dgl.DGLGraph,
        rep1_cdims: torch.Tensor,
        degrees: list[int],
        basis: torch.Tensor,
        edge_weights: torch.Tensor,
        node_features: torch.Tensor,
    ) -> torch.Tensor:

        U, V = graph.edges()

        tmp = matmulStepOne(
            basis,
            node_features,
            U,
            rep1_cdims,
        )

        return matmulStepTwo(
            tmp,
            edge_weights,
            degrees,
        )


class EquivariantMatmulKernel(nn.Module):
    """
    Implements the matmul step in an equivariant transformer with a custom
    kernel backend for the forward and backward passes.
    """

    def __init__(
        self: EquivariantMatmulKernel,
        repr: ProductRepr,
    ) -> None:

        super().__init__()
        self.register_buffer(
            'rep1_cdims',
            torch.tensor(repr.rep1.cumdims())
        )
        self.degrees = repr.rep2.lvals
        self.n = repr.rep2.mult
        self._kernel = _EquivariantMatmulKernel().apply

    def forward(
        self: EquivariantMatmulKernel,
        graph: dgl.DGLGraph,
        basis: torch.Tensor,
        edge_weights: torch.Tensor,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the kernel.
        """

        # Apply the kernel
        return self._kernel(
            graph,
            self.rep1_cdims,
            self.degrees,
            basis,
            edge_weights,
            node_features,
        )
