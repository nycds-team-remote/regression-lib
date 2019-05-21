from utils import make_mat, mat_mul, transpose
from sklearn.linear_model import LinearRegression
from typing import List
from pytest import approx

import numpy as np

# def mat_mul(mat1, mat2):


def test_transpose() -> None:

    assert transpose([]) == []

    assert transpose([[1, 1], [0, 0]]) == [[1, 0], [1, 0]]

    input = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    expected = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

    assert transpose(input) == expected


def test_mat_mul():

    assert mat_mul([[1, 2]], [[5], [6]]) == [[17]]

    assert mat_mul([[0, 1, 2], [0, 1, 2]], [[0, 1], [0, 1], [0, 1]]) == [
        [0, 3],
        [0, 3],
    ]

    assert mat_mul([[10, 200], [3000, 40000]], [[0, 1], [2, 3]]) == [
        [400, 610],
        [80000, 123000],
    ]

    # list(np.matmul([[10, 200], [3000, 40000]], [[0, 1], [2, 3]]))
    #
    # list(np.matmul([[1, 2]], [[0], [1]]))


np.matmul([[1, 2]], [[5], [6]])

[list(range(3))] * 2
[list(range(2))] * 3


np.matmul([list(range(3))] * 2, [list(range(2))] * 3)
