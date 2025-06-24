#include "eigen-3.4.0/Eigen/Dense"
#include "eigen-3.4.0/Eigen/QR"
#include <algorithm>
#include <cctype>
#include <execution>
#include <future>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
using namespace Eigen;

// This is for the binding to python later
// If you want to compile just in C++, comment that out!
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <pybind11/complex.h>
// #include <pybind11/functional.h>
// #include <pybind11/chrono.h>
// #include <stdexcept>
// namespace py = pybind11;



// TODO: remove eigen lib.

class Block_Binv {
public:
  unordered_map<int, MatrixXd> blocks;
  unordered_map<int, MatrixXd> pblocks;
  MatrixXd pMatrix;
  MatrixXd matrixToInverse;

private:
  int block_size = 1;
  int matrix_cols = 0;
  int matrix_rows = 0;

protected:
  void divide_into_blocks(const MatrixXd& matrix, int blocks_amount = 0) {
    // initialization
    matrix_cols = matrix.cols();
    matrix_rows = matrix.rows();
    pMatrix.resize(matrix_cols, matrix_rows);
    matrixToInverse = MatrixXd::Identity(matrix_cols, matrix_cols);

    // dividing into blocks
    if (blocks_amount == 0)
      blocks_amount = matrix_cols;

    block_size = round((float)matrix_cols / (float)blocks_amount);
    int start = 0;
    for (int b = 0; b < blocks_amount; b++) {
      int col_amount = block_size;
      if (b == blocks_amount - 1)
        col_amount = matrix_cols - start;
      blocks[b] = matrix.middleCols(start, col_amount);
      start = start + block_size;
      }
    }

protected:
  void multuplyPseudoinversedAndFillMatrixToInverse(MatrixXd pblock,
    int block_id,
    int anotherBlockId) {
    auto product = pblock * blocks[anotherBlockId];
    int row_start = block_id * block_size;
    int col_start = anotherBlockId * block_size;
    matrixToInverse.block(row_start, col_start, product.rows(),
      product.cols()) = product;
    }

protected:
  void formMatrices(int block_id) {
    auto pblock = blocks[block_id].completeOrthogonalDecomposition().pseudoInverse();

    int start = block_id * block_size;
    pMatrix.block(start, 0, pblock.rows(), pblock.cols()) = pblock;

    vector<future<void>> futures;
    for (const auto [id, block] : blocks) {
      if (block_id != id) {
        futures.push_back(
          async(launch::async,
            &Block_Binv::multuplyPseudoinversedAndFillMatrixToInverse,
            this, pblock, block_id, id));
        }
      }
    for (auto& task : futures)
      task.wait(); // maybe not needed?
    }

  void pinvblocks() {
    vector<future<void>> futures;
    for (const auto [id, block] : blocks) {
      futures.push_back(
        // TODO: optimize: remove the second launch. to it right from the first parallelization.
        async(launch::async, &Block_Binv::formMatrices, this, id));
      }
    for (auto& task : futures)
      task.wait(); // maybe not needed?
    }

public:
  MatrixXd binv(const MatrixXd& matrix, int block_amount = 2) {
    divide_into_blocks(matrix, block_amount);
    pinvblocks();
    return matrixToInverse.inverse() * pMatrix;
    }
  };

// PYBIND11_MODULE(example, m) {
//     m.doc() = "Binv function"; // optional module docstring
//     m.def("add", &add, "This is the docstring for the add function");
//     py::class_<Block_Binv>(m, "Block_Binv")
//         .def(py::init<const bool &>(),py::arg("verbose") = true) //
//         .def("divide_into_blocks", &Block_Binv::divide_into_blocks)
//         .def("pinvblock",          &Block_Binv::pinvblock)
//         .def("pinvblocks",         &Block_Binv::pinvblocks)
//         .def("form_row",           &Block_Binv::form_row)
//         .def("build_Pmatrix",      &Block_Binv::build_Pmatrix)
//         .def("binv",               &Block_Binv::binv);
// }

// You see the version of c++ in -std=c++11 and you can change it to 14,17,20
// c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes)
// binv_cpp.cpp -o binv_cpp$(python3-config --extension-suffix)