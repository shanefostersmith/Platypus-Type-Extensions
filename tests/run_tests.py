import sys
import pytest

def main(): 
    TESTS = [ # ordered to compile the numba methods first and avoid speed tests
        "tests/test_real_int/test_real_methods.py",
        "tests/test_real_int/test_integer_methods.py",
        "tests/test_list_ranges/test_list_ranges.py",
        "tests/test_bins_sets/test_bins_sets.py",
        "tests/test_core/"
    ]
    sys.exit(pytest.main(TESTS))
 
if __name__ == "__main__":
    main()  