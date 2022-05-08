# utilities.py
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

import numpy as np
import os
from numba import njit
from sys import platform
if platform == "linux" or platform == "linux2":
    import resource


def identify_mu_for_snr(x, T, eta, xi, v_el, prot):
    """"
    :param x: The desired SNR to be achieved.
    :param T:
    :param eta:
    :param xi:
    :param v_el:
    :param prot:
    :return: The modulation variance that achieves the desired SNR.
    """

    if prot == "homodyne":
        Chi = xi + (1 + v_el) / (T * eta)
    elif prot == "heterodyne":
        Chi = xi + (2 + v_el) / (T * eta)
    else:
        raise ValueError("Wrong protocol given.")

    return x * Chi + 1


def percent_to_length_conversion(x, a):
    """
    Converts the length of an optical fiber to channel losses.
    :param x: The transmissivity of the optical channel.
    :param a: The attenuation of the optical fiber.
    :return: The fiber length (in km).
    """

    return -10 * np.log10(x) / a


def length_to_percent_conversion(x, a):
    """
    Converts the length of an optical fiber to channel losses.
    :param x: The length of the optical fiber.
    :param a: The attenuation of the optical fiber.
    :return: The channel losses (in decibel).
    """

    return 10 ** (-a * x / 10)


def percent_to_decibel_conversion(x):
    return 10 * np.log10(x)


def decibel_to_percent_conversion(x):
    return 10 ** (x / 10)


def q_ary_to_binary(m, q):
    """
    Converts a q-ary sequence into a binary sequence of length q.
    :param m: The q-ary sequence.
    :param q: The Galois field exponent.
    :return: The binary representations of the q-ary sequences.
    """

    mA_bin = np.empty(len(m) * q, dtype=np.int8)  # Binary representation of Alice's q-ary message
    for i in range(len(m)):
        bitsA = np.binary_repr(m[i], width=q)
        for j in range(q):
            mA_bin[i * q + j] = bitsA[j]
    return mA_bin


@njit(parallel=True, fastmath=True, cache=True)
def row_rotation(i, array_in):
    """
    Moves a row to the bottom of the array, while moving all others one index up.
    """

    array_out = array_in.copy()
    num_rows = array_out.shape[0]
    # Move the row to the bottom
    array_out[num_rows - 1] = array_in[i, :]
    # Rotate the bottom rows upwards
    index = 2
    while (num_rows - index) >= i:
        array_out[num_rows - index, :] = array_in[num_rows - index + 1]
        index = index + 1
    return array_out


def peak_memory_measurement(proc):
    """"
    Measures the peak memory consumption of the software in MiB the until a certain runtime point.
    :param proc: The current process.
    """

    if os.name == "nt":  # Works only in Windows systems
        mem = proc.memory_full_info().peak_wset / 2 ** 20  # Original measurement in bytes
    else:
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Original measurement in KiB
    return mem



