#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <future>
#include <thread>
#include <mutex>
#include <queue>
#include <functional>
#include <condition_variable>
#include <chrono>

// Optimized matrix class with memory-efficient operations
class Matrix {
public:
    std::vector<double> data;
    int rows, cols;
    
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}
    Matrix(int r, int c, const std::vector<double>& d) : rows(r), cols(c), data(d) {}
    
    // Inline accessors for better performance
    inline double& operator()(int i, int j) { return data[i * cols + j]; }
    inline const double& operator()(int i, int j) const { return data[i * cols + j]; }
    
    // Standard transpose for clarity
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
    
    // Cache-friendly matrix multiplication with blocking
    Matrix multiply(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        
        Matrix result(rows, other.cols);
        const int block_size = 64;
        
        // Initialize result to zero
        std::fill(result.data.begin(), result.data.end(), 0.0);
        
        // Blocked multiplication for better cache locality
        for (int ii = 0; ii < rows; ii += block_size) {
            for (int jj = 0; jj < other.cols; jj += block_size) {
                for (int kk = 0; kk < cols; kk += block_size) {
                    int i_max = std::min(ii + block_size, rows);
                    int j_max = std::min(jj + block_size, other.cols);
                    int k_max = std::min(kk + block_size, cols);
                    
                    for (int i = ii; i < i_max; i++) {
                        for (int j = jj; j < j_max; j++) {
                            double sum = 0.0;
                            for (int k = kk; k < k_max; k++) {
                                sum += (*this)(i, k) * other(k, j);
                            }
                            result(i, j) += sum;
                        }
                    }
                }
            }
        }
        return result;
    }
    
    // Get block copy (for compatibility with original algorithm)
    Matrix getBlock(int start_row, int start_col, int block_rows, int block_cols) const {
        Matrix block(block_rows, block_cols);
        for (int i = 0; i < block_rows; i++) {
            for (int j = 0; j < block_cols; j++) {
                if (start_row + i < rows && start_col + j < cols) {
                    block(i, j) = (*this)(start_row + i, start_col + j);
                }
            }
        }
        return block;
    }
    
    // Set block
    void setBlock(int start_row, int start_col, const Matrix& block) {
        for (int i = 0; i < block.rows; i++) {
            for (int j = 0; j < block.cols; j++) {
                if (start_row + i < rows && start_col + j < cols) {
                    (*this)(start_row + i, start_col + j) = block(i, j);
                }
            }
        }
    }
    
    static Matrix identity(int size) {
        Matrix id(size, size);
        for (int i = 0; i < size; i++) {
            id(i, i) = 1.0;
        }
        return id;
    }
    
    void print() const {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
};

// Optimized SVD with better numerical stability
class SVD {
public:
    static Matrix pseudoInverse(const Matrix& A, double tolerance = 1e-12) {
        if (A.rows >= A.cols) {
            // Tall matrix: A+ = (A^T * A)^-1 * A^T
            Matrix AT = A.transpose();
            Matrix ATA = AT.multiply(A);
            Matrix inv_ATA = invertMatrix(ATA, tolerance);
            return inv_ATA.multiply(AT);
        } else {
            // Wide matrix: A+ = A^T * (A * A^T)^-1
            Matrix AT = A.transpose();
            Matrix AAT = A.multiply(AT);
            Matrix inv_AAT = invertMatrix(AAT, tolerance);
            return AT.multiply(inv_AAT);
        }
    }
    
private:
    // Optimized matrix inversion with partial pivoting
    static Matrix invertMatrix(const Matrix& A, double tolerance) {
        int n = A.rows;
        if (n != A.cols) {
            throw std::invalid_argument("Matrix must be square for inversion");
        }
        
        // Create augmented matrix [A | I]
        Matrix augmented(n, 2 * n);
        
        // Initialize augmented matrix
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented(i, j) = A(i, j);
                augmented(i, j + n) = (i == j) ? 1.0 : 0.0;
            }
        }
        
        // Gauss-Jordan elimination with partial pivoting
        for (int i = 0; i < n; i++) {
            // Find pivot
            int pivot_row = i;
            double max_val = std::abs(augmented(i, i));
            
            for (int k = i + 1; k < n; k++) {
                double val = std::abs(augmented(k, i));
                if (val > max_val) {
                    max_val = val;
                    pivot_row = k;
                }
            }
            
            // Swap rows if needed
            if (pivot_row != i) {
                for (int j = 0; j < 2 * n; j++) {
                    std::swap(augmented(i, j), augmented(pivot_row, j));
                }
            }
            
            // Check for singularity
            if (max_val < tolerance) {
                throw std::runtime_error("Matrix is singular or nearly singular");
            }
            
            // Scale pivot row
            double pivot = augmented(i, i);
            double inv_pivot = 1.0 / pivot;
            for (int j = 0; j < 2 * n; j++) {
                augmented(i, j) *= inv_pivot;
            }
            
            // Eliminate column
            for (int k = 0; k < n; k++) {
                if (k != i && std::abs(augmented(k, i)) > tolerance) {
                    double factor = augmented(k, i);
                    for (int j = 0; j < 2 * n; j++) {
                        augmented(k, j) -= factor * augmented(i, j);
                    }
                }
            }
        }
        
        // Extract inverse matrix
        Matrix inverse(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse(i, j) = augmented(i, j + n);
            }
        }
        
        return inverse;
    }
};

class OptimizedCpuBlockBinv {
private:
    std::vector<Matrix> blocks;
    std::vector<Matrix> pblocks;
    Matrix pMatrix;
    Matrix matrixToInverse;
    int block_size;
    int matrix_cols;
    int matrix_rows;
    int num_blocks;

public:
    OptimizedCpuBlockBinv() : pMatrix(1, 1), matrixToInverse(1, 1) {}
    
private:
    void divideIntoBlocks(const Matrix& matrix, int blocks_amount) {
        matrix_cols = matrix.cols;
        matrix_rows = matrix.rows;
        num_blocks = blocks_amount;
        
        if (blocks_amount == 0) {
            blocks_amount = std::max(2, std::min(8, matrix_cols / 32));
            num_blocks = blocks_amount;
        }
        
        block_size = matrix_cols / blocks_amount;
        int remaining = matrix_cols % blocks_amount;
        
        // Initialize matrices
        pMatrix = Matrix(matrix_cols, matrix_rows);
        matrixToInverse = Matrix::identity(matrix_cols);
        
        // Clear previous blocks
        blocks.clear();
        pblocks.clear();
        blocks.reserve(num_blocks);
        pblocks.reserve(num_blocks);
        
        // Divide into blocks with proper handling of remainder
        int start = 0;
        for (int b = 0; b < blocks_amount; b++) {
            int col_amount = block_size + (b < remaining ? 1 : 0);
            
            Matrix block = matrix.getBlock(0, start, matrix_rows, col_amount);
            blocks.push_back(std::move(block));
            
            // Reserve space for pseudoinverse block
            pblocks.push_back(Matrix(col_amount, matrix_rows));
            
            start += col_amount;
        }
    }
    
    void multiplyPseudoInverseAndFillMatrix(const Matrix& pblock, int block_id, int another_block_id) {
        Matrix product = pblock.multiply(blocks[another_block_id]);
        
        // Calculate proper row and column starts for variable block sizes
        int row_start = 0;
        int col_start = 0;
        
        for (int i = 0; i < block_id; i++) {
            row_start += blocks[i].cols; // Use cols because we're placing in identity matrix
        }
        for (int i = 0; i < another_block_id; i++) {
            col_start += blocks[i].cols;
        }
        
        matrixToInverse.setBlock(row_start, col_start, product);
    }
    
    void formMatrices(int block_id) {
        // Compute pseudoinverse for this block
        Matrix pblock = SVD::pseudoInverse(blocks[block_id]);
        pblocks[block_id] = pblock;
        
        // Set block in pMatrix - calculate proper start position
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
        // Process blocks in parallel
        std::vector<std::future<void>> futures;
        futures.reserve(num_blocks);
        
        for (int id = 0; id < num_blocks; id++) {
            futures.push_back(
                std::async(std::launch::async, &OptimizedCpuBlockBinv::formMatrices, this, id)
            );
        }
        
        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
    }

public:
    Matrix binv(const Matrix& matrix, int block_amount = 0) {
        if (block_amount == 0) {
            // Auto-determine optimal block count
            int max_threads = std::thread::hardware_concurrency();
            block_amount = std::min(max_threads, std::max(2, matrix.cols / 64));
        }
        
        // Ensure we don't have more blocks than columns
        block_amount = std::min(block_amount, matrix.cols);
        
        divideIntoBlocks(matrix, block_amount);
        processAllBlocks();
        
        // Final computation
        try {
            Matrix inv_matrixToInverse = SVD::pseudoInverse(matrixToInverse);
            return inv_matrixToInverse.multiply(pMatrix);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Using pseudoinverse for final computation due to: " << e.what() << std::endl;
            Matrix inv_matrixToInverse = SVD::pseudoInverse(matrixToInverse);
            return inv_matrixToInverse.multiply(pMatrix);
        }
    }
    
    // Convenience method for vector input
    std::vector<double> binv(const std::vector<double>& matrix_data, 
                           int rows, int cols, int block_amount = 0) {
        if (matrix_data.size() != static_cast<size_t>(rows * cols)) {
            throw std::invalid_argument("Matrix size doesn't match dimensions");
        }
        
        Matrix matrix(rows, cols, matrix_data);
        Matrix result = binv(matrix, block_amount);
        
        return result.data;
    }
};

// Example usage with performance measurement
void example_usage() {
    try {
        OptimizedCpuBlockBinv cpu_binv;
        
        // Test with both original small example and larger matrix
        std::cout << "=== Testing with 4x4 matrix ===" << std::endl;
        std::vector<double> small_matrix = {
            1.0, 2.0, 3.0, 4.0,
            2.0, 3.0, 4.0, 1.0,
            3.0, 4.0, 1.0, 2.0,
            4.0, 1.0, 2.0, 3.0
        };
        
        auto result_small = cpu_binv.binv(small_matrix, 4, 4, 2);
        
        std::cout << "Input matrix:" << std::endl;
        Matrix input_small(4, 4, small_matrix);
        input_small.print();
        
        std::cout << "\nResult matrix:" << std::endl;
        Matrix result_small_matrix(4, 4, result_small);
        result_small_matrix.print();
        
        std::cout << "\nVerification (should be close to identity):" << std::endl;
        Matrix verification_small = input_small.multiply(result_small_matrix);
        verification_small.print();
        
        // Test with larger matrix
        std::cout << "\n=== Testing with 6x6 Hilbert matrix ===" << std::endl;
        const int size = 6;
        std::vector<double> matrix_data;
        matrix_data.reserve(size * size);
        
        // Generate Hilbert matrix (well-conditioned for this size)
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                matrix_data.push_back(1.0 / (i + j + 1.0));
            }
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = cpu_binv.binv(matrix_data, size, size, 3);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Input matrix (" << size << "x" << size << "):" << std::endl;
        Matrix input(size, size, matrix_data);
        input.print();
        
        std::cout << "\nResult matrix:" << std::endl;
        Matrix result_matrix(size, size, result);
        result_matrix.print();
        
        // Verify by multiplying original * result
        std::cout << "\nVerification (should be close to identity):" << std::endl;
        Matrix verification = input.multiply(result_matrix);
        verification.print();
        
        std::cout << "\nComputation time: " << duration.count() << " microseconds" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main() {
    example_usage();
    return 0;
}

// Compile with: g++ -std=c++14 -O3 -march=native -pthread -flto optimized_cpu_block_binv.cpp -o optimized_cpu_block_binv