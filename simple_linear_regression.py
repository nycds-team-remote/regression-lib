from typing import NamedTuple, List
import numpy as np


class Point(NamedTuple):
    x: float
    y: float


class SimpleLinearModel(NamedTuple):
    # also known as beta0 or b
    intercept: float
    # also known as beta1, slope or m
    coef: float


# similar to construct and then fit
def get_simple_linear_model(points: List[Point]) -> SimpleLinearModel:

    N = len(points)

    xySum = sum([p.x * p.y for p in points])

    xSum = sum([p.x for p in points])

    x2Sum = sum([p.x ** 2 for p in points])

    ySum = sum([p.y for p in points])

    coef = ((N * xySum) - (xSum * ySum)) / ((N * x2Sum) - (xSum ** 2))

    intercept: float = ((x2Sum * ySum) - (xSum * xySum)) / (
        (N * x2Sum) - (xSum ** 2)
    )

    return SimpleLinearModel(intercept, coef)


def predict1(model: SimpleLinearModel, x: float) -> float:
    return model.intercept + model.coef * x


def predict(model: SimpleLinearModel, xs: List[float]) -> List[float]:
    return [predict1(model, x) for x in xs]


def regression_sum_squares(
    model: SimpleLinearModel, points: List[Point]
) -> float:
    return sum(
        ((model.intercept + model.coef * p.x) - p.y) ** 2 for p in points
    )


def residual_sum_squares(ys: List[float]) -> float:
    mean: float = np.average(ys)
    return sum([(y - mean) ** 2 for y in ys])


def score(model: SimpleLinearModel, points: List[Point]) -> float:
    u = regression_sum_squares(model, points)
    v = residual_sum_squares([p.y for p in points])
    return 1 - u / v
