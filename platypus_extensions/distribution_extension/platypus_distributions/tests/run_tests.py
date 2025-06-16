import sys
import pytest

def main(): 
    TESTS = [
        "platypus_distributions/tests/test_bounds.py",
        "platypus_distributions/tests/test_distributions.py",
        "platypus_distributions/tests/test_symmetric.py",
        "platypus_distributions/tests/test_cache.py::test_lru_smoke",
    ]
    sys.exit(pytest.main(TESTS))
 
if __name__ == "__main__":
    main()  