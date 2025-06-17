import os
import sys
import pytest

def main(): 
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    TESTS = []
    for test_dir in [ # ordered to compile the numba methods first and avoid speed tests
        "test_real_int/test_real_methods.py",
        "test_real_int/test_integer_methods.py",
        "test_list_ranges/test_list_ranges.py",
        "test_bins_sets/test_bins_sets.py",
        "test_core/"
    ]:
        TESTS.append(os.path.join(project_root, "tests", test_dir))
    sys.exit(pytest.main(TESTS))
 
if __name__ == "__main__":
    main()  