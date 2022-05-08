# maths.py
# Copyright 2020 Alexandros Georgios Mountogiannakis

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import utilities
from numba import njit, prange


def convolution_theorem(x, y):
    conv = np.fft.ifft(np.multiply(np.fft.fft(x).astype(np.complex64),
                                   np.fft.fft(y).astype(np.complex64))).astype(np.complex64)
    return conv


def shannon_entropy(x, n_bks, n):
    _, counts_y = np.unique(x.ravel(), return_counts=True)
    p = counts_y / (n_bks * n)  # Appearance probabilities of the discretized values
    h = -np.sum(p * np.log2(p))  # Shannon entropy of the discretized values of the key generation states
    return h


def von_neumann_entropy(v):
    """
    A bosonic entropic function which calculates the Von Neumann entropy.
    :param v: The symplectic eigenvalue.
    :return: The von Neumann entropy.
    """

    return ((v + 1) / 2) * np.log2((v + 1) / 2) - ((v - 1) / 2) * np.log2((v - 1) / 2)


def symplectic_eigenvalue_calculation(V):
    """
    If the matrix V is a 4 × 4 positive-definite matrix, it can be expressed in the block form [[A C], [C^T B]]. In such
    a case, the symplectic spectrum can be calculated by the formula for ν±. Alternatively, the eigenvalues are found
    from the modulus |iΩV|.
    :param V The matrix whose symplectic eigenvalues must be obtained.
    :return The symplectic eigenvalues of V.
    """

    if V.shape == (2, 2):
        iOmega = np.array([[0, 1j], [-1j, 0]])  # iΩ matrix
        v = np.linalg.eigvals((np.dot(iOmega, V)))
        v_1 = np.abs(v[0])

        assert v_1 > 1  # Assert that the eigenvalue is positive and larger than unit
        return v_1

    elif V.shape == (4, 4):
        # Check if the given matrix is 4 x 4, symmetric and positive definite
        iOmega = np.array([[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, -1j, 0]])  # iΩ matrix
        v = np.linalg.eigvals((np.dot(iOmega, V)))  # Find the real eigenvalues
        v_abs = np.abs(v).astype(np.float32)  # Take the absolute value and truncate the precision for np.unique to work
        v_unique, v_counts = np.unique(v_abs, return_counts=True)  # Keep only the two unique values from the eigenvalue matrix
        if v_counts[0] == 4:  # If both eigenvalues are the same
            v_1 = v_unique[0]
            v_2 = v_unique[0]
        else:
            v_1 = v_unique[0]
            v_2 = v_unique[1]

        # Assert that the eigenvalues are positive and larger than unit
        assert v_1 > 1
        assert v_2 > 1

        return v_1, v_2


@njit(fastmath=True, cache=True)
def conditional_probability(k, i, r, a, p, d):
    """
    Calculates the conditional probability to be used for the calculation of the a priori probabilities.
    :param k: The discretized variable.
    :param i: The value of the bin.
    :param r: The correlation parameter.
    :param a: The discretization cut-off parameter.
    :param p: The number of bins exponent.
    :param d: The constant-size interval divider.
    :return: The conditional probability P(K|X).
    """

    if i == 0:
        ak = -np.inf
        bk = -a + d
    elif i == 2 ** p - 1:
        ak = -a + (2 ** p - 1) * d
        bk = np.inf
    else:
        ak = -a + i * d
        bk = -a + (i + 1) * d

    A = (ak - k * r) / np.sqrt(2 * (1 - r ** 2))
    B = (bk - k * r) / np.sqrt(2 * (1 - r ** 2))
    prob = 0.5 * (math.erf(B) - math.erf(A))

    return prob


@njit(fastmath=True, parallel=True, cache=True)
def gaussian_elimination(h, n, m):
    """
    Finds the systematic form of a given matrix. The systematic form of the matrix is [I | P], where I is the identity
    matrix and P is the parity matrix. Gaussian Elimination over GF(2) is performed, which implies that the code retains
    the same abilities, as elementary row and column operations are performed. A pseudocode can be found at "A Fast
    Algorithm for Gaussian Elimination over GF(2) and its Implementation on the GAPP".
    :param h: The matrix of the code to be converted to a systematic form.
    :param n: The number of columns of the matrix.
    :param m: The number of rows of the matrix.
    :return arr: The matrix in reduced row echelon (systematic) form.
    """

    threshold = m
    i = 0
    while i < threshold:
        found = False  # Flag indicating that the row contains a non-zero entry (pivot)
        # Find the leftmost nonzero for every column including and after the current row i
        for j in prange(i, n):
            if h[i][j] == 1:
                found = True
                temp = h[:, i]
                h[:, i] = h[:, j]
                h[:, j] = temp  # Column swap so that (i,i) = 1
                break
        if found:
            for u in prange(0, m):
                if u == i:
                    continue
                # Checking for 1's
                if h[u][i] == 1:
                    for v in prange(0, m):
                        h[u][v] = (h[u][v] + h[i][v]) % 2  # Add row i to row v modulus 2
            # All the entries above & below (i, i) are now 0
            i = i + 1
        else:
            # print("No 1 was found at row i =", i, ". Row will be moved to the bottom.")
            # print(np.roll(code, i + 1, axis=0))  # The axis argument is not yet supported by Numba
            h = utilities.row_rotation(i, h)  # Bring the row full of zeros to the bottom and move the bottom row upwards
            threshold -= 1  # The threshold should be now be now reduced

    arr = h[0: i, :]
    return arr


@njit()
def orthogonal_matrix(n):
    """"
    The orthogonal matrices of size d × d which have been provided in the Appendix of Ref. Leverrier et al, 2008.
    """

    if n == 2:
        A1 = np.array([[1, 0], [0, 1]])
        A2 = np.array([[0, -1], [1, 0]])
        return [A1, A2]
    elif n == 4:
        A1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        A2 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
        A3 = np.array([[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]])
        A4 = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]])
        return [A1, A2, A3, A4]
    elif n == 8:
        A1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])
        A2 = np.array([[0, -1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1, 0]])
        A3 = np.array([[0, 0, -1, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0]])
        A4 = np.array([[0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0]])
        A5 = np.array([[0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, -1], [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]])
        A6 = np.array([[0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1, 0], [0, -1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]])
        A7 = np.array([[0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0], [0, 0, -1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0, 0, 0]])
        A8 = np.array([[0, 0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]])
        return [A1, A2, A3, A4, A5, A6, A7, A8]
    else:
        raise ValueError("n not applicable")


@njit(cache=True)
def symplectic_transformation(v, w):
    """
    Symplectic (conjugate transpose) operation for covariance matrices
    :param v:
    :param w:
    :return:
    """

    return v @ w @ np.transpose(np.conjugate(v))





