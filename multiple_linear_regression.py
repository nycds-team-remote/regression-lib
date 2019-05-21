from typing import NamedTuple, List
import numpy as np


# def get_r2(features: List[List[float]], target: List[float],
# betas:List[float]):


class MultipleLinearModel(NamedTuple):
    betas: List[float]


def transpose(matrix):
    return list(map(list, zip(*matrix)))


transpose([[1, 1], [0, 0]])
# def get_multiple_linear_model(
#     X: List[List[float]], y: List[float]
# ) -> List[float]:
#     xT = transpose(X)
#     left = inverse(mat_mul(xT, X))
