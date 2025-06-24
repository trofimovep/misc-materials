#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <future>
#include <thread>

// Simple matrix class for CPU computation
class Matrix {
public:
    std::vector<double> data;
    int rows, cols;
    
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}
    Matrix(int r, int c, const std::vector<double>& d) : rows(r), cols(c), data(d) {}
    
    double& operator()(int i, int j) { return data[i * cols + j]; }
    const double& operator()(int i, int j) const { return data[i * cols + j]; }
    
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
    
    Matrix multiply(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < cols; k++) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
    
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

// Simple SVD implementation for pseudoinverse
class SVD {
public:
    static Matrix pseudoInverse(const Matrix& A, double tolerance = 1e-10) {
        // Simplified pseudoinverse using Moore-Penrose method
        // For a complete implementation, you'd want to use a proper SVD library like LAPACK
        
        Matrix AT = A.transpose();
        Matrix ATA = AT.multiply(A);
        Matrix AAT = A.multiply(AT);
        
        // For now, we'll use a simplified approach
        // In practice, you'd compute the actual SVD decomposition
        
        if (A.rows >= A.cols) {
            // Tall matrix: A+ = (A^T * A)^-1 * A^T
            Matrix inv_ATA = invertMatrix(ATA, tolerance);
            return inv_ATA.multiply(AT);
        } else {
            // Wide matrix: A+ = A^T * (A * A^T)^-1
            Matrix inv_AAT = invertMatrix(AAT, tolerance);
            return AT.multiply(inv_AAT);
        }
    }
    
private:
    static Matrix invertMatrix(const Matrix& A, double tolerance) {
        // Simplified matrix inversion using Gauss-Jordan elimination
        int n = A.rows;
        if (n != A.cols) {
            throw std::invalid_argument("Matrix must be square for inversion");
        }
        
        // Create augmented matrix [A | I]
        Matrix augmented(n, 2 * n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented(i, j) = A(i, j);
                augmented(i, j + n) = (i == j) ? 1.0 : 0.0;
            }
        }
        
        // Gauss-Jordan elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            int pivot_row = i;
            for (int k = i + 1; k < n; k++) {
                if (std::abs(augmented(k, i)) > std::abs(augmented(pivot_row, i))) {
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
            if (std::abs(augmented(i, i)) < tolerance) {
                throw std::runtime_error("Matrix is singular or nearly singular");
            }
            
            // Scale pivot row
            double pivot = augmented(i, i);
            for (int j = 0; j < 2 * n; j++) {
                augmented(i, j) /= pivot;
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

class CpuBlockBinv {
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
    CpuBlockBinv() : pMatrix(1, 1), matrixToInverse(1, 1) {}
    
private:
    void divideIntoBlocks(const Matrix& matrix, int blocks_amount) {
        matrix_cols = matrix.cols;
        matrix_rows = matrix.rows;
        num_blocks = blocks_amount;
        
        if (blocks_amount == 0) {
            blocks_amount = matrix_cols;
            num_blocks = blocks_amount;
        }
        
        block_size = std::round((float)matrix_cols / (float)blocks_amount);
        
        // Initialize matrices
        pMatrix = Matrix(matrix_cols, matrix_rows);
        matrixToInverse = Matrix::identity(matrix_cols);
        
        // Clear previous blocks
        blocks.clear();
        pblocks.clear();
        
        // Divide into blocks
        int start = 0;
        for (int b = 0; b < blocks_amount; b++) {
            int col_amount = block_size;
            if (b == blocks_amount - 1) {
                col_amount = matrix_cols - start;
            }
            
            Matrix block = matrix.getBlock(0, start, matrix_rows, col_amount);
            blocks.push_back(block);
            
            // Reserve space for pseudoinverse block
            pblocks.push_back(Matrix(col_amount, matrix_rows));
            
            start += block_size;
        }
    }
    
    void multiplyPseudoInverseAndFillMatrix(const Matrix& pblock, int block_id, int another_block_id) {
        Matrix product = pblock.multiply(blocks[another_block_id]);
        
        int row_start = block_id * block_size;
        int col_start = another_block_id * block_size;
        
        matrixToInverse.setBlock(row_start, col_start, product);
    }
    
    void formMatrices(int block_id) {
        // Compute pseudoinverse for this block
        Matrix pblock = SVD::pseudoInverse(blocks[block_id]);
        pblocks[block_id] = pblock;
        
        // Set block in pMatrix
        int start = block_id * block_size;
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
        
        for (int id = 0; id < num_blocks; id++) {
            futures.push_back(
                std::async(std::launch::async, &CpuBlockBinv::formMatrices, this, id)
            );
        }
        
        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
    }

public:
    Matrix binv(const Matrix& matrix, int block_amount = 2) {
        divideIntoBlocks(matrix, block_amount);
        processAllBlocks();
        
        // Final computation: matrixToInverse^-1 * pMatrix
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
                           int rows, int cols, int block_amount = 2) {
        if (matrix_data.size() != rows * cols) {
            throw std::invalid_argument("Matrix size doesn't match dimensions");
        }
        
        Matrix matrix(rows, cols, matrix_data);
        Matrix result = binv(matrix, block_amount);
        
        return result.data;
    }
};

// Example usage function
void example_usage() {
    try {
        CpuBlockBinv cpu_binv;
        
        // Example 4x4 matrix
        std::vector<double> matrix_data = {
            1.0, 2.0, 3.0, 4.0,
            2.0, 3.0, 4.0, 1.0,
            3.0, 4.0, 1.0, 2.0,
            4.0, 1.0, 2.0, 3.0
        };
        
        auto result = cpu_binv.binv(matrix_data, 4, 4, 2);
        
        std::cout << "Input matrix:" << std::endl;
        Matrix input(4, 4, matrix_data);
        input.print();
        
        std::cout << "\nResult matrix:" << std::endl;
        Matrix result_matrix(4, 4, result);
        result_matrix.print();
        
        // Verify by multiplying original * result
        std::cout << "\nVerification (should be close to identity):" << std::endl;
        Matrix verification = input.multiply(result_matrix);
        verification.print();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main() {
    example_usage();
    return 0;
}

// g++ -std=c++14 -O3 -pthread cpu_block_binv.cpp -o cpu_block_binv