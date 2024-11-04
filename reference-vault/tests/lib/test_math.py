import pytest
import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray
from lib.math import sigmoid

KNOWN_SIGMOID_VALUES: List[Tuple[float, float]] = [
    (0.0, 0.5),
    (1.0, 0.7310585786300049),
    (-1.0, 0.2689414213699951),
    (4.0, 0.9820137900379085),
    (-4.0, 0.0179862099620915),
    (np.inf, 1.0),
    (-np.inf, 0.0),
]


@pytest.mark.parametrize(
    "x, expected",
    [(np.array([x]), np.array([y])) for x, y in KNOWN_SIGMOID_VALUES],
)
def test_sigmoid_known_values(
    x: NDArray[np.float64], expected: NDArray[np.float64]
) -> None:
    assert np.allclose(sigmoid(x), expected)


@pytest.mark.parametrize(
    "m, n",
    [
        (5, 5),
        (1, 5),
        (5, 1),
        (3, 5),
        (5, 3),
    ],
)
def test_sigmoid_for_shapes_of_known_values(m: int, n: int) -> None:
    # Randomly select m*n pairs from known values
    indices = np.random.choice(len(KNOWN_SIGMOID_VALUES), size=m * n, replace=True)
    x_values = np.array([KNOWN_SIGMOID_VALUES[i][0] for i in indices])
    expected_values = np.array([KNOWN_SIGMOID_VALUES[i][1] for i in indices])

    # Reshape to requested dimensions
    x = x_values.reshape(m, n)
    expected = expected_values.reshape(m, n)

    assert np.allclose(sigmoid(x), expected)


@pytest.mark.parametrize(
    "invalid_input",
    [
        pytest.param("string", id="string"),
        pytest.param(1, id="plain_int"),
        pytest.param([1.0], id="plain_list"),
        pytest.param(np.array(["string"]), id="string_array"),
        pytest.param(np.array([1], dtype=np.int32), id="int32_array"),
    ],
)
def test_sigmoid_type_errors(invalid_input: object) -> None:
    with pytest.raises(TypeError):
        sigmoid(invalid_input)  # type: ignore
