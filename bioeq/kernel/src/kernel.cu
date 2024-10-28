#include <assert.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <torch/extension.h>
#include <vector>

using std::vector, std::min;

inline void checkCudaError() {
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

constexpr int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

constexpr int64_t TILE_SIZE = 32;
constexpr int64_t N_IN_TILE_SIZE = 32;
constexpr int64_t MAX_DEGREE = 6;

template <typename scalar_t>
__global__ void flashEquivarianceStepOneKernel(
    const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits>
        basis,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        features,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits>
        src,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits>
        in_cdims,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> output,
    int64_t NREPS) {

  // basis shape: (E_MAX, D_IN_MAX, L_MAX, D_OUT_MAX)
  // features shape: (V_MAX, N_IN_MAX, D_IN_MAX)
  // src shape: (E_MAX,)
  // out_blocks shape: (L_MAX + 1,)
  // in_blocks shape: (L_MAX + 1,)
  // output shape: (E_MAX, L_TMP_MAX, D_OUT_MAX, N_IN_MAX)

  int64_t E_MAX = output.size(0);
  int64_t N_IN_MAX = features.size(1);
  int64_t D_IN_MAX = features.size(2);
  int64_t D_OUT_MAX = basis.size(3);
  int64_t L_MAX = basis.size(2);

  int64_t TID_X = threadIdx.x;
  int64_t TID_Y = threadIdx.y;

  // TODO: all of this needs to be re-written with non-contiguous degrees in
  // mind

  // edge index
  int64_t E = blockIdx.x;
  // d_out index
  int64_t D_OUT = TID_X;
  // n_in index
  int64_t N_IN = blockIdx.y * N_IN_TILE_SIZE + TID_Y;

  // Get the source node of this edge
  int64_t U_IX = src[E];

  // Allocate shared memory
  __shared__ scalar_t shared_features[16][N_IN_TILE_SIZE];
  __shared__ scalar_t shared_basis[16][16];
  // Load the features into shared memory
  if (N_IN < N_IN_MAX) {
    shared_features[TID_X][TID_Y] = features[U_IX][N_IN][TID_X];
  } else {
    shared_features[TID_X][TID_Y] = scalar_t(0.0);
  }
  __syncthreads();

  scalar_t tmps[MAX_DEGREE] = {0.0};
  int64_t COUNT = 0;

  for (int64_t L = 0; L < L_MAX; L++) {

    int64_t L_BLOCK = (L + 1) / 2;

    // load the basis into shared memory
    // what the fuck is going on here? what happens if D_OUT_MAX is smaller
    // than D_IN_MAX
    // should consider doing tiling over D_IN, but this is not so easy to
    // write up (although if we know the representation dimensions at
    // compile time, this should make it easier)

    // if D_OUT is small but L is large, then the basis is zero, so maybe
    // we can speed things up that way? Or at least we can save on memory.
    __syncthreads();
    if (TID_Y < D_IN_MAX) {
      shared_basis[TID_Y][TID_X] = basis[E][TID_Y][L][D_OUT];
    } else if (TID_Y < D_OUT_MAX) {
      shared_basis[TID_Y][TID_X] = scalar_t(0.0);
    }
    __syncthreads();

    // Loop over the input blocks
    for (int64_t IN_BL = L_BLOCK; IN_BL < NREPS; IN_BL++) {
      // Zero out the output
      tmps[IN_BL] = scalar_t(0.0);
      // Matrix multiplication over the input degrees
      int64_t D_IN_LOWER = in_cdims[IN_BL];
      int64_t D_IN_UPPER = in_cdims[IN_BL + 1];
      for (int64_t D_IN = D_IN_LOWER; D_IN < D_IN_UPPER; D_IN++) {
        tmps[IN_BL] += shared_features[D_IN][TID_Y] * shared_basis[D_IN][TID_X];
      }
    }

    // Write the output
    if (N_IN < N_IN_MAX) {
      for (int64_t IN_BL = L_BLOCK; IN_BL < NREPS; IN_BL++) {
        output[E][(COUNT + IN_BL - L_BLOCK) * 32 + N_IN][D_OUT] = tmps[IN_BL];
      }
      COUNT += NREPS - L_BLOCK;
    }
  }
}

torch::Tensor matmulStepOneCUDA(torch::Tensor basis, torch::Tensor features,
                                torch::Tensor src, torch::Tensor in_blocks) {

  // Get the dimensions of the output tensor
  int64_t E_MAX = basis.size(0);
  int64_t D_OUT_MAX = basis.size(3);
  int64_t NREPS = in_blocks.size(0) - 1;
  int64_t N_IN_MAX = features.size(1);
  int64_t D_IN_MAX = features.size(2);
  // Get the scalar type of the output tensor
  c10::ScalarType sc_t = features.scalar_type();
  // Allocate memory for the output tensor
  torch::Tensor output = torch::empty(
      {E_MAX, NREPS * NREPS * N_IN_MAX, D_OUT_MAX}, torch::CUDA(sc_t));

  dim3 blockDim, gridDim;
  // The number of threads in each thread block
  blockDim.x = D_OUT_MAX;
  blockDim.y = N_IN_TILE_SIZE;
  // The number of thread blocks in each grid
  gridDim.x = E_MAX;
  gridDim.y = ceilDiv(N_IN_MAX, N_IN_TILE_SIZE);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      sc_t, "flashEquivarianceStepOneKernel", ([&] {
        flashEquivarianceStepOneKernel<scalar_t><<<
            gridDim, blockDim, 0, c10::cuda::getCurrentCUDAStream()>>>(
            basis.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
            features.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            src.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            in_blocks.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            output.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            NREPS);
      }));

  checkCudaError();
  return output;
}

template <typename scalar_t>
__global__ void flashEquivarianceStepTwoKernel(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        inter,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        rw,
    int64_t OFFSET, int64_t IX_OFFSET, int64_t IX_MAX,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        output) {

  // inter shape: (E_MAX, L_TMP_MAX, D_OUT_MAX, N_IN_MAX)
  // rw shape: (E_MAX, N_OUT_MAX, N_IN_MAX * IX_MAX)
  // output shape: (E_MAX, N_OUT_MAX, D_OUT_MAX)

  int64_t MULT_OUT_MAX = output.size(1);

  int64_t TID_X = threadIdx.x;
  int64_t TID_Y = threadIdx.y;

  // edge index
  int64_t EDGE = blockIdx.x;
  // n_out index
  int64_t MULT_OUT = blockIdx.y * TILE_SIZE + TID_X;
  // repr_out index
  int64_t REPR_OUT = TID_Y + OFFSET;
  // REPR_DIM
  int64_t REPR_DIM = blockDim.y;

  extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_tmp_char[];
  scalar_t *shared_tmp = reinterpret_cast<scalar_t *>(shared_tmp_char);
  // Initialize the output value
  scalar_t tmp = scalar_t(0.0);

  for (int64_t IX_BLOCK = 0; IX_BLOCK < IX_MAX; IX_BLOCK += TILE_SIZE) {
    // Load the tmp features into shared memory
    shared_tmp[TID_X * REPR_DIM + TID_Y] =
        inter[EDGE][IX_BLOCK + TID_X][REPR_OUT];
    __syncthreads();
    // Accumulate over this block
    for (int64_t IX = 0; IX < TILE_SIZE; IX++) {
      tmp += shared_tmp[IX * REPR_DIM + TID_Y] *
             rw[EDGE][IX + IX_BLOCK + IX_OFFSET][MULT_OUT];
    }
  }
  // Write the output
  if (MULT_OUT < MULT_OUT_MAX) {
    output[EDGE][MULT_OUT][REPR_OUT] = tmp;
  }
}

torch::Tensor matmulStepTwoCUDA(torch::Tensor inter, torch::Tensor rw,
                                vector<int64_t> degrees) {

  // Get the dimensions of the output tensor
  int64_t EDGE_MAX = inter.size(0);
  int64_t REPR_OUT_MAX = inter.size(2);
  int64_t PROD_REPR_MAX = rw.size(1);
  int64_t MULT_IN_MAX = rw.size(2);
  int64_t MULT_OUT_MAX = rw.size(3);
  // Get the scalar type of the output tensor
  c10::ScalarType sc_t = inter.scalar_type();
  // Allocate memory for the output tensor
  torch::Tensor output =
      torch::empty({EDGE_MAX, MULT_OUT_MAX, REPR_OUT_MAX}, torch::CUDA(sc_t));
  // Reshape the edge weights for multiplication
  rw = rw.view({EDGE_MAX, PROD_REPR_MAX * MULT_IN_MAX, MULT_OUT_MAX});

  dim3 gridDim, blockDim;
  gridDim.x = EDGE_MAX;
  gridDim.y = ceilDiv(MULT_OUT_MAX, TILE_SIZE);
  blockDim.x = TILE_SIZE;

  int64_t NREPS = degrees.size();

  int64_t OFFSET = 0;
  int64_t IX_OFFSET = 0;
  int64_t IX_MAX;
  for (const int64_t &deg : degrees) {
    int64_t dim = 2 * deg + 1;

    IX_MAX = 0;
    for (const int64_t &deg2 : degrees) {
      int64_t dim2 = 2 * deg2 + 1;
      IX_MAX += min(dim, dim2);
    }
    IX_MAX *= MULT_IN_MAX;
    printf("IX_MAX; dim: %d %d\n", IX_MAX, dim);

    // Get the size of the shared memory required
    int64_t sharedMemSize = dim * TILE_SIZE * torch::elementSize(sc_t);
    // Set the number of threads for processing the REPR dimension
    blockDim.y = dim;
    // Launch the kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        sc_t, "flashEquivarianceStepTwoKernel", ([&] {
          flashEquivarianceStepTwoKernel<
              scalar_t><<<gridDim, blockDim, sharedMemSize,
                          c10::cuda::getCurrentCUDAStream()>>>(
              inter.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
              rw.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
              OFFSET, IX_OFFSET, IX_MAX,
              output
                  .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
        }));
    OFFSET += dim;
    IX_OFFSET += IX_MAX;
  }

  checkCudaError();
  return output;
}
