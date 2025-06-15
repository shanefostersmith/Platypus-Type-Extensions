import sys
import pytest

def main(): 
    TESTS = [
        "custom_types/distributions/tests/test_bounds.py",
        "custom_types/distributions/tests/test_distributions.py",
        "custom_types/distributions/tests/test_symmetric.py",
        "custom_types/distributions/tests/test_cache.py::test_lru_smoke",
    ]
    sys.exit(pytest.main(TESTS))
 
if __name__ == "__main__":
    main()  