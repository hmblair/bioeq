#include "kittens.cuh"

using namespace kittens;

// Use 4 warps per block
constexpr int NUM_WORKERS = 4;
// Rows per worker tile
template<int D> constexpr size_t ROWS = 16*(128/D);
template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, ROWS<D>, D, L>;
template<int D, typename T=float> using attn_tile = rt<T, ROWS<D>, ROWS<D>>;
template<int D> using shared_tile = st_bf<ROWS<D>, D>;
// EDGE, M, N
template<int D> using global_layout = gl<bf16, -1, -1, D>;
template<int D> struct globals { global_layout<D> Xg, Wg, Og; };
