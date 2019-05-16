from pytest import approx
from typing import List
from sklearn.linear_model import LinearRegression

from simple_linear_regression import (
    Point,
    SimpleLinearModel,
    get_simple_linear_model,
    score,
)


def test_get_simple_linear_model_simple() -> None:
    points = [Point(0, 1), Point(1, 1.5), Point(2, 2), Point(3, 2.5)]

    result = get_simple_linear_model(points)

    expected = SimpleLinearModel(intercept=1, coef=0.5)

    assert result == expected

    assert score(result, points) == 1.0


def test_get_simple_linear_model_against_sklearn() -> None:
    xs = [4.1, 6.5, 12.6, 25.5, 29.8, 38.6, 46, 52.8, 59.6, 66.3, 74.7]
    ys = [2.2, 4.5, 10.4, 23.1, 27.9, 36.8, 44.3, 50.7, 57.5, 64.1, 72.6]

    x_matrix = [[x] for x in xs]

    points: List[Point] = [Point(p[0], p[1]) for p in zip(xs, ys)]

    sk_result = LinearRegression().fit(x_matrix, ys)

    result: SimpleLinearModel = get_simple_linear_model(points)

    assert result.coef == approx(sk_result.coef_[0])

    assert result.intercept == approx(sk_result.intercept_)

    assert score(result, points) == approx(sk_result.score(x_matrix, ys))
