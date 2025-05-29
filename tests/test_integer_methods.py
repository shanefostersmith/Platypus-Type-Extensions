from conftest import *
import numpy as np
import pytest
from platypus.tests.test_types import Integer
from custom_types.utils import int_to_gray_encoding, gray_encoding_to_int
from custom_types.integer_methods.integer_methods import multi_int_crossover, single_binary_swap

MIN_VALUE = -1
MAX_VALUE = 16
def test_static_int_gray_conversion():
    int_type = Integer(MIN_VALUE, MAX_VALUE)
    for i in range(MIN_VALUE, MAX_VALUE + 1):
        to_gray = int_to_gray_encoding(i, MIN_VALUE, MAX_VALUE)
        assert list(to_gray) == int_type.encode(i)

        to_int = gray_encoding_to_int(MIN_VALUE, MAX_VALUE, to_gray, int)
        assert to_int == i


@pytest.mark.parametrize(
    "nrows, ncols, zero_reference, zero_column",
    [
        (2, 5, False, False),
        (2, 2, True,  False),
        (2, 2, False, True),
        (2, 1, False, True),
        (0, 1, False, True),
    ],
    ids=["rand","zero_ref","zero_col","only_zero", "zero_by_one"],
    indirect=["nrows","ncols","zero_reference","zero_column"],
)
def test_multi_crossover(
    nrows, ncols, zero_reference, zero_column,
    np_normalized_matrix32,
    noffspring
):
    bool_matrix = np_normalized_matrix32.astype(dtype = np.bool_)
    print(f"bool matrix: {bool_matrix}")
    result = multi_int_crossover(bool_matrix, noffspring)
    print(f"result = {result}, noffspring = {noffspring}")
    
    assert result.shape[0] == noffspring

@pytest.mark.parametrize("vec_length", [1, 2, 5], ids=lambda L: f"length={L}")
@pytest.mark.parametrize(
        "nrows, ncols, zero_reference, zero_column",
        [
            (2, 5, False, False),
            (2, 2, True,  False),
            (1, 2, False, True),
            (2, 1, False, True),
            (0, 1, False, True),
        ],
        ids=["rand","zero_ref","zero_col","only_zero","zero_by_one"],
        indirect=["nrows","ncols","zero_reference","zero_column"],
    )
def test_binary_swap(
    nrows, ncols, zero_reference, zero_column,
    np_normalized_matrix32,
    vec_length):
    
    bool_matrix = np_normalized_matrix32.astype(dtype = np.bool_)
    offspring_vector = np.random.randint(0, 2, size=vec_length).astype(dtype = np.bool_)
    if bool_matrix.shape[1] != vec_length:
        with pytest.raises(ValueError):
            single_binary_swap(bool_matrix, offspring_vector)
    else:
        result = single_binary_swap(bool_matrix, offspring_vector)
    
