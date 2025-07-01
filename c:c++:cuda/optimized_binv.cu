#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

// CUDA error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << status << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "cuSOLVER error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << status << std::endl; \
            exit(1); \
        } \
    } while(0)

// CUDA kernels for matrix operations
__global__ void transposeKernel(const double* input, double* output, 
                               int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < cols && idy < rows) {
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}

__global__ void setIdentityKernel(double* matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < size && idy < size) {
        matrix[idy * size + idx] = (idx == idy) ? 1.0 : 0.0;
    }
}

__global__ void copyBlockKernel(const double* source, double* dest,
                               int src_rows, int src_cols,
                               int dest_rows, int dest_cols,
                               int src_start_row, int src_start_col,
                               int dest_start_row, int dest_start_col,
                               int block_rows, int block_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < block_cols && idy < block_rows) {
        int src_row = src_start_row + idy;
        int src_col = src_start_col + idx;
        int dest_row = dest_start_row + idy;
        int dest_col = dest_start_col + idx;
        
        if (src_row < src_rows && src_col < src_cols &&
            dest_row < dest_rows && dest_col < dest_cols) {
            dest[dest_row * dest_cols + dest_col] = 
                source[src_row * src_cols + src_col];
        }
    }
}

// GPU Matrix class with CUDA operations
class GpuMatrix {
public:
    thrust::device_vector<double> data;
    int rows, cols;
    
    GpuMatrix(int r, int c) : rows(r), cols(c), data(r * c) {}
    
    GpuMatrix(int r, int c, const std::vector<double>& host_data) 
        : rows(r), cols(c), data(host_data) {}
    
    // Copy constructor
    GpuMatrix(const GpuMatrix& other) 
        : rows(other.rows), cols(other.cols), data(other.data) {}
    
    // Assignment operator
    GpuMatrix& operator=(const GpuMatrix& other) {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            data = other.data;
        }
        return *this;
    }
    
    double* ptr() { return thrust::raw_pointer_cast(data.data()); }
    const double* ptr() const { return thrust::raw_pointer_cast(data.data()); }
    
    // Get submatrix (creates a copy)
    GpuMatrix getBlock(int start_row, int start_col, int block_rows, int block_cols) const {
        GpuMatrix result(block_rows, block_cols);
        
        dim3 blockSize(16, 16);
        dim3 gridSize((block_cols + blockSize.x - 1) / blockSize.x,
                      (block_rows + blockSize.y - 1) / blockSize.y);
        
        copyBlockKernel<<<gridSize, blockSize>>>(
            ptr(), result.ptr(),
            rows, cols, block_rows, block_cols,
            start_row, start_col, 0, 0,
            block_rows, block_cols
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        return result;
    }
    
    // Set block
    void setBlock(int start_row, int start_col, const GpuMatrix& block) {
        dim3 blockSize(16, 16);
        dim3 gridSize((block.cols + blockSize.x - 1) / blockSize.x,
                      (block.rows + blockSize.y - 1) / blockSize.y);
        
        copyBlockKernel<<<gridSize, blockSize>>>(
            block.ptr(), ptr(),
            block.rows, block.cols, rows, cols,
            0, 0, start_row, start_col,
            block.rows, block.cols
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Convert to host vector
    std::vector<double> toHost() const {
        thrust::host_vector<double> host_data = data;
        return std::vector<double>(host_data.begin(), host_data.end());
    }
    
    void print() const {
        auto host_data = toHost();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << host_data[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

// CUDA-accelerated linear algebra operations
class CudaLinAlg {
private:
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    
public:
    CudaLinAlg() {
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    }
    
    ~CudaLinAlg() {
        cublasDestroy(cublas_handle);
        cusolverDnDestroy(cusolver_handle);
    }
    
    // Accessor methods for handles
    cublasHandle_t getCublasHandle() { return cublas_handle; }
    cusolverDnHandle_t getCusolverHandle() { return cusolver_handle; }
    
    // Matrix multiplication using cuBLAS
    GpuMatrix multiply(const GpuMatrix& A, const GpuMatrix& B) const {
        if (A.cols != B.rows) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        
        GpuMatrix C(A.rows, B.cols);
        const double alpha = 1.0, beta = 0.0;
        
        CUBLAS_CHECK(cublasDgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            B.cols, A.rows, A.cols,
            &alpha,
            B.ptr(), B.cols,
            A.ptr(), A.cols,
            &beta,
            C.ptr(), C.cols));
        
        return C;
    }
    
    // Matrix transpose
    GpuMatrix transpose(const GpuMatrix& A) const {
        GpuMatrix AT(A.cols, A.rows);
        
        dim3 blockSize(16, 16);
        dim3 gridSize((A.cols + blockSize.x - 1) / blockSize.x,
                      (A.rows + blockSize.y - 1) / blockSize.y);
        
        transposeKernel<<<gridSize, blockSize>>>(
            A.ptr(), AT.ptr(), A.rows, A.cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        return AT;
    }
    
    // SVD-based pseudoinverse using cuSOLVER
    GpuMatrix pseudoInverse(const GpuMatrix& A, double tolerance = 1e-12) const {
        int m = A.rows;
        int n = A.cols;
        int min_mn = std::min(m, n);
        
        // Copy input matrix
        GpuMatrix A_copy = A;
        
        // Allocate arrays for SVD
        thrust::device_vector<double> S(min_mn);
        thrust::device_vector<double> U(m * m);
        thrust::device_vector<double> VT(n * n);
        thrust::device_vector<int> devInfo(1);
        
        // Query workspace size
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(
            cusolver_handle, m, n, &lwork));
        
        thrust::device_vector<double> work(lwork);
        
        // Compute SVD: A = U * S * V^T
        CUSOLVER_CHECK(cusolverDnDgesvd(
            cusolver_handle, 'A', 'A',
            m, n,
            A_copy.ptr(), m,
            thrust::raw_pointer_cast(S.data()),
            thrust::raw_pointer_cast(U.data()), m,
            thrust::raw_pointer_cast(VT.data()), n,
            thrust::raw_pointer_cast(work.data()), lwork,
            nullptr,
            thrust::raw_pointer_cast(devInfo.data())));
        
        // Check if SVD succeeded
        thrust::host_vector<int> h_info = devInfo;
        if (h_info[0] != 0) {
            throw std::runtime_error("SVD failed");
        }
        
        // Compute pseudoinverse: A+ = V * S^+ * U^T
        // where S^+ has 1/s_i for s_i > tolerance, 0 otherwise
        
        // Create S_pinv matrix
        GpuMatrix S_pinv(n, m);
        thrust::fill(S_pinv.data.begin(), S_pinv.data.end(), 0.0);
        
        // Copy singular values and invert them
        thrust::host_vector<double> h_S = S;
        for (int i = 0; i < min_mn; i++) {
            if (h_S[i] > tolerance) {
                // Set S_pinv[i][i] = 1/S[i]
                thrust::device_reference<double> ref = S_pinv.data[i * m + i];
                ref = 1.0 / h_S[i];
            }
        }
        
        // Compute A+ = V * S^+ * U^T
        GpuMatrix V(n, n);
        GpuMatrix UT(m, m);
        
        // Transpose VT to get V and U to get UT
        V = transpose(GpuMatrix(n, n, std::vector<double>(VT.begin(), VT.end())));
        UT = transpose(GpuMatrix(m, m, std::vector<double>(U.begin(), U.end())));
        
        // First: V * S^+
        GpuMatrix temp = multiply(V, S_pinv);
        // Then: (V * S^+) * U^T
        return multiply(temp, UT);
    }
    
    // Create identity matrix
    static GpuMatrix identity(int size) {
        GpuMatrix I(size, size);
        
        dim3 blockSize(16, 16);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x,
                      (size + blockSize.y - 1) / blockSize.y);
        
        setIdentityKernel<<<gridSize, blockSize>>>(I.ptr(), size);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        return I;
    }
};

// CUDA-accelerated block-based pseudoinverse
class CudaBlockBinv {
private:
    std::vector<GpuMatrix> blocks;
    std::vector<GpuMatrix> pblocks;
    GpuMatrix pMatrix;
    GpuMatrix matrixToInverse;
    std::unique_ptr<CudaLinAlg> linalg;
    int block_size;
    int matrix_cols;
    int matrix_rows;
    int num_blocks;
    
    // CUDA streams for parallel processing
    std::vector<cudaStream_t> streams;

public:
    CudaBlockBinv() : pMatrix(1, 1), matrixToInverse(1, 1) {
        linalg = std::make_unique<CudaLinAlg>();
        
        // Create CUDA streams for parallel processing
        int num_streams = 4;
        streams.resize(num_streams);
        for (int i = 0; i < num_streams; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }
    }
    
    ~CudaBlockBinv() {
        for (auto& stream : streams) {
            cudaStreamDestroy(stream);
        }
    }
    
private:
    void divideIntoBlocks(const GpuMatrix& matrix, int blocks_amount) {
        matrix_cols = matrix.cols;
        matrix_rows = matrix.rows;
        num_blocks = blocks_amount;
        
        if (blocks_amount == 0) {
            // Auto-determine block count based on GPU memory and matrix size
            size_t free_mem, total_mem;
            CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
            
            // Use heuristic: aim for blocks that fit well in GPU memory
            int optimal_blocks = std::max(2, std::min(8, matrix_cols / 64));
            blocks_amount = optimal_blocks;
            num_blocks = blocks_amount;
        }
        
        block_size = matrix_cols / blocks_amount;
        int remaining = matrix_cols % blocks_amount;
        
        // Initialize matrices
        pMatrix = GpuMatrix(matrix_cols, matrix_rows);
        matrixToInverse = CudaLinAlg::identity(matrix_cols);
        
        // Clear previous blocks
        blocks.clear();
        pblocks.clear();
        blocks.reserve(num_blocks);
        pblocks.reserve(num_blocks);
        
        // Divide into blocks
        int start = 0;
        for (int b = 0; b < blocks_amount; b++) {
            int col_amount = block_size + (b < remaining ? 1 : 0);
            
            GpuMatrix block = matrix.getBlock(0, start, matrix_rows, col_amount);
            blocks.push_back(std::move(block));
            
            // Pre-allocate pseudoinverse block
            pblocks.emplace_back(col_amount, matrix_rows);
            
            start += col_amount;
        }
    }
    
    void multiplyPseudoInverseAndFillMatrix(const GpuMatrix& pblock, 
                                          int block_id, int another_block_id) {
        GpuMatrix product = linalg->multiply(pblock, blocks[another_block_id]);
        
        // Calculate proper positions
        int row_start = 0, col_start = 0;
        for (int i = 0; i < block_id; i++) {
            row_start += blocks[i].cols;
        }
        for (int i = 0; i < another_block_id; i++) {
            col_start += blocks[i].cols;
        }
        
        matrixToInverse.setBlock(row_start, col_start, product);
    }
    
    void formMatrices(int block_id) {
        // Compute pseudoinverse for this block
        GpuMatrix pblock = linalg->pseudoInverse(blocks[block_id]);
        pblocks[block_id] = pblock;
        
        // Set block in pMatrix
        int start = 0;
        for (int i = 0; i < block_id; i++) {
            start += blocks[i].cols;
        }
        pMatrix.setBlock(start, 0, pblock);
        
        // Multiply with other blocks
        for (int id = 0; id < num_blocks; id++) {
            if (block_id != id) {
                multiplyPseudoInverseAndFillMatrix(pblock, block_id, id);
            }
        }
    }
    
    void processAllBlocks() {
        // Process blocks sequentially 
        // Each CUDA operation within formMatrices is inherently parallel
        for (int id = 0; id < num_blocks; id++) {
            formMatrices(id);
        }
        
        // Ensure all GPU operations are complete
        CUDA_CHECK(cudaDeviceSynchronize());
    }

public:
    GpuMatrix binv(const GpuMatrix& matrix, int block_amount = 0) {
        if (block_amount == 0) {
            // Auto-determine based on GPU capabilities
            int device;
            cudaGetDevice(&device);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);
            
            int max_blocks = prop.multiProcessorCount;
            block_amount = std::min(max_blocks / 4, 
                                  std::max(2, matrix.cols / 128));
        }
        
        block_amount = std::min(block_amount, matrix.cols);
        
        divideIntoBlocks(matrix, block_amount);
        processAllBlocks();
        
        // Final computation
        try {
            GpuMatrix inv_matrixToInverse = linalg->pseudoInverse(matrixToInverse);
            return linalg->multiply(inv_matrixToInverse, pMatrix);
        } catch (const std::exception& e) {
            std::cerr << "Warning: " << e.what() << std::endl;
            return linalg->pseudoInverse(matrixToInverse);
        }
    }
    
    // Convenience method for host data
    std::vector<double> binv(const std::vector<double>& matrix_data, 
                           int rows, int cols, int block_amount = 0) {
        if (matrix_data.size() != static_cast<size_t>(rows * cols)) {
            throw std::invalid_argument("Matrix size doesn't match dimensions");
        }
        
        GpuMatrix matrix(rows, cols, matrix_data);
        GpuMatrix result = binv(matrix, block_amount);
        
        return result.toHost();
    }
};

// Example usage
void cuda_example_usage() {
    try {
        // Check CUDA device
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            std::cerr << "No CUDA devices found!" << std::endl;
            return;
        }
        
        std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
        
        CudaBlockBinv cuda_binv;
        
        // Test with 4x4 matrix
        std::cout << "=== CUDA Testing with 4x4 matrix ===" << std::endl;
        std::vector<double> matrix_data = {
            1.0, 2.0, 3.0, 4.0,
            2.0, 3.0, 4.0, 1.0,
            3.0, 4.0, 1.0, 2.0,
            4.0, 1.0, 2.0, 3.0
        };
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = cuda_binv.binv(matrix_data, 4, 4, 2);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Input matrix:" << std::endl;
        GpuMatrix input(4, 4, matrix_data);
        input.print();
        
        std::cout << "\nResult matrix:" << std::endl;
        GpuMatrix result_matrix(4, 4, result);
        result_matrix.print();
        
        std::cout << "\nCUDA computation time: " << duration.count() << " microseconds" << std::endl;
        
        // Test larger matrix for performance comparison
        std::cout << "\n=== CUDA Testing with 16x16 matrix ===" << std::endl;
        const int size = 16;
        std::vector<double> large_matrix(size * size);
        
        // Generate a test matrix
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                large_matrix[i * size + j] = 1.0 / (i + j + 1.0) + 0.1 * (i == j ? 1.0 : 0.0);
            }
        }
        
        start = std::chrono::high_resolution_clock::now();
        auto large_result = cuda_binv.binv(large_matrix, size, size, 4);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Large matrix computation time: " << duration.count() << " microseconds" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main() {
    cuda_example_usage();
    return 0;
}

// Compile with: nvcc -std=c++14 -O3 -lcublas -lcusolver cuda_block_binv.cu -o cuda_block_binv
