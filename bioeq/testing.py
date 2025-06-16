from __future__ import annotations
import torch
import dgl
from bioeq.geom import (
    Irrep,
    Repr,
)
from bioeq.modules import EquivariantTransformer
from bioeq.polymer import Polymer, GeometricPolymer
from time import time
from torch.cuda import Event
import torch.nn as nn


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
N_ITER = 25

BATCH_SIZE = 8
ATOMS_PER_MOL = 200
NUM_ATOMS = BATCH_SIZE * ATOMS_PER_MOL

NODE_DEGREE = 16
IN_SIZE = 32
HIDDEN_SIZE = 32
OUT_SIZE = 32
EDGE_DIM = 1
NHEADS = 8
NLAYERS = 0
EDGE_HIDDEN_DIM = 32
ATTN_DROPOUT = 0.0
DROPOUT = 0.0

in_lvals = [0]
hidden_lvals = [0, 1, 2]
out_lvals = [0, 1]

x = torch.randn(BATCH_SIZE, ATOMS_PER_MOL, 3, device=device, dtype=dtype)
g = dgl.knn_graph(x, NODE_DEGREE).to(device)
x = x.view(-1, 3)
U, V = g.edges()

in_rep = Repr(in_lvals, IN_SIZE)
out_rep = Repr(out_lvals, OUT_SIZE)
hidden_rep = Repr(hidden_lvals, HIDDEN_SIZE)

f = torch.randn(NUM_ATOMS, IN_SIZE, in_rep.dim(), device=device, dtype=dtype)
edge_feats = torch.linalg.norm(x[U] - x[V], dim=-1)[:, None]

residue_sizes = torch.tensor([NUM_ATOMS]).long()
chain_sizes = torch.tensor([NUM_ATOMS]).long()
molecule_sizes = torch.tensor([NUM_ATOMS]).long()

elements = torch.randint(0, 5, (f.size(0),))
residues = torch.randint(0, 5, (f.size(0),))
atom_names = torch.randint(0, 5, (f.size(0),))

embedding = nn.Embedding(5, IN_SIZE)

polymer = Polymer(
    coordinates=x,
    elements=elements,
    residues=residues,
    atom_names=atom_names,
    graph=g,
    residue_sizes=residue_sizes,
    chain_sizes=chain_sizes,
    molecule_sizes=molecule_sizes,
)
geom_polymer = GeometricPolymer(
    polymer=polymer,
    node_features=embedding(elements)[..., None],
    edge_features=edge_feats,
)

rotaxis = torch.randn(3)
rotaxis /= rotaxis.norm()
rotang = (torch.randn(1) * torch.pi)
rot1 = in_rep.rot(rotaxis, rotang).to(device).to(dtype).squeeze(0)
rot2 = out_rep.rot(rotaxis, rotang).to(device).to(dtype).squeeze(0)
rot3d = Repr().rot(rotaxis, rotang).to(device).to(dtype).squeeze(0)
x_r = torch.einsum('ij,bi->bj', rot3d, x)
f_r = torch.einsum('ij,bmi->bmj', rot1, f)

geom_polymer_r = geom_polymer.rotate(rot3d)

eqtransformer = EquivariantTransformer(
    in_repr=in_rep,
    out_repr=out_rep,
    hidden_repr=hidden_rep,
    hidden_layers=NLAYERS,
    edge_dim=EDGE_DIM,
    edge_hidden_dim=EDGE_HIDDEN_DIM,
    nheads=NHEADS,
    dropout=DROPOUT,
    attn_dropout=ATTN_DROPOUT,
).to(device).to(dtype)

out = eqtransformer.polymer(geom_polymer)
if torch.cuda.is_available():
    start = Event(enable_timing=True)
    end = Event(enable_timing=True)
    start.record()
else:
    start = time()
for _ in range(N_ITER):
    out = eqtransformer.polymer(geom_polymer)
if torch.cuda.is_available():
    torch.cuda.synchronize()
    end.record()
    t_elapsed = start.elapsed_time(end)
else:
    t_elapsed = (time() - start) * 1000
print(f"Time elapsed: {t_elapsed / N_ITER:.2f} ms")

out.node_features = torch.einsum('ij,bmi->bmj', rot2, out.node_features)

out2 = eqtransformer.polymer(geom_polymer_r)

diff = (out.node_features - out2.node_features).abs().max()
print(out.node_features.shape)
print(f'Error: {diff:.7f}')
breakpoint()
