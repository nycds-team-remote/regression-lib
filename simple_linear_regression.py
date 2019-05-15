from typing import NamedTuple, List


class Point(NamedTuple):
    x: float
    y: float


class SimpleLinearModel(NamedTuple):
    # also known as beta0 or b
    intercept: float
    # also known as beta1, slope or m
    coef: float


def get_simple_linear_model(points: List[Point]) -> SimpleLinearModel:

    N = len(points)

    xySum: float = sum([p.x * p.y for p in points])

    xSum: float = sum([p.x for p in points])

    x2Sum: float = sum([p.x ** 2 for p in points])

    ySum: float = sum([p.y for p in points])

    coef = ((N * xySum) - (xSum * ySum)) / ((N * x2Sum) - (xSum ** 2))

    intercept = ((x2Sum * ySum) - (xSum * xySum)) / ((N * x2Sum) - (xSum ** 2))

    return SimpleLinearModel(intercept, coef)


# def score(model: SimpleLinearModel, points: List[Point])
