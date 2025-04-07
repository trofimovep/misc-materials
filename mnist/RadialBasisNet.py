# https://en.wikipedia.org/wiki/Radial_basis_function_network
# The original work one can check in the file: sources/Tapson Learning the pseudoinverse solution to network weights.pdf
# The previous implementation in the file: Rbf_Implementations.ipynb


import numpy as np
from Method import Method


device = "cpu"

if device == "cpu":
    import numpy as np
    from numpy.linalg import inv, pinv, norm

    print("Choose device: cpu")
elif device == "cuda":
    import cupy as np
    from cupy.linalg import inv, pinv
else:
    raise ValueError("Idiot!")


import time

# from numba import njit
import random

# import torch


class RBF:
    def __init__(
        self,
        input_vector_size: int,
        hidden_neurons_amount: int,
        output_vector_size: int,
    ):
        # each row in a layer (matrix) is the neuron weights
        self.output = None
        self.hidden_output = None
        self.hidden_layer = (
            2 * np.random.rand(hidden_neurons_amount, input_vector_size) - 1
        )
        self.output_layer = (
            2 * np.random.rand(output_vector_size, hidden_neurons_amount) - 1
        )

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def hidden_output_calculation(self, input_vec):
        if np.isscalar(np.asarray(input_vec)):
            self.hidden_output = self.sigmoid(self.hidden_layer * input_vec)
        else:
            self.hidden_output = self.sigmoid(
                np.matmul(self.hidden_layer, np.asarray(input_vec))
            )
        return self.hidden_output

    def feedforward(self, input_vec: np.ndarray):
        self.hidden_output = self.hidden_output_calculation(input_vec=input_vec)
        self.output = np.matmul(np.transpose(self.output_layer), self.hidden_output)
        return self.output

    def fast_binv(self, C, split_index=None, method="svd"):
        """
        Optimized pseudoinverse of [A | B] using QR/SVD block preprocessing.

        Parameters:
            C (ndarray): Input matrix of shape (m, n1 + n2)
            split_index (int or None): Index to split C into A and B (columns). If None, split in half.
            method (str): 'svd' (default) or 'qr'

        Returns:
            AB_pinv (ndarray): Pseudoinverse of the concatenated matrix [A | B]
        """
        m, n = C.shape
        if split_index is None:
            split_index = n // 2

        A = C[:, :split_index]
        B = C[:, split_index:]

        n1 = A.shape[1]
        n2 = B.shape[1]

        if method == "svd":
            UA, SA, VA_T = np.linalg.svd(A, full_matrices=False)
            UB, SB, VB_T = np.linalg.svd(B, full_matrices=False)

            RA_inv = VA_T.T * (1.0 / SA)
            RB_inv = VB_T.T * (1.0 / SB)

            # Pad the R blocks to same column size
            RA_inv_padded = np.hstack((RA_inv, np.zeros((n1, n2))))
            RB_inv_padded = np.hstack((np.zeros((n2, n1)), RB_inv))

        elif method == "qr":
            UA, RA = np.linalg.qr(A, mode="reduced")
            UB, RB = np.linalg.qr(B, mode="reduced")

            RA_inv = np.linalg.inv(RA)
            RB_inv = np.linalg.inv(RB)

            RA_inv_padded = np.hstack((RA_inv, np.zeros((n1, n2))))
            RB_inv_padded = np.hstack((np.zeros((n2, n1)), RB_inv))

        else:
            raise ValueError("Method must be 'svd' or 'qr'.")

        Q_comb = np.hstack((UA, UB))
        R_comb = np.vstack((RA_inv_padded, RB_inv_padded))  # shape (n1 + n2, n1 + n2)

        AB_pinv = R_comb @ Q_comb.T  # Final pseudoinverse
        return AB_pinv

    def train(
        self,
        train_set: np.ndarray,
        train_out_set: np.ndarray,
        method: Method = Method.BINV,
    ):
        train_hidden_out = np.vstack(
            [self.hidden_output_calculation(train) for train in train_set]
        )
        if method == Method.BINV:
            pinv_train_hidden_out = self.fast_binv(train_hidden_out, method="qr")
        elif method == Method.PINV:
            pinv_train_hidden_out = np.linalg.pinv(train_hidden_out)
        else:
            Exception("Idiot!")
        self.output_layer = np.matmul(pinv_train_hidden_out, train_out_set)
