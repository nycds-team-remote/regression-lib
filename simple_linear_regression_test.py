from typing import List

from simple_linear_regression import (
    Point,
    SimpleLinearModel,
    get_simple_linear_model,
)


def test_get_simple_linear_model() -> None:
    xs = [4.1, 6.5, 12.6, 25.5, 29.8, 38.6, 46, 52.8, 59.6, 66.3, 74.7]
    ys = [2.2, 4.5, 10.4, 23.1, 27.9, 36.8, 44.3, 50.7, 57.5, 64.1, 72.6]

    points: List[Point] = [Point(p[0], p[1]) for p in zip(xs, ys)]

    result = get_simple_linear_model(points)

    expected = SimpleLinearModel(
        intercept=-2.0066596290477032, coef=0.9992155004070224
    )

    assert result == expected
