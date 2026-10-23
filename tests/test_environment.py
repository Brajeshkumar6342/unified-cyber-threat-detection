
import sys

import numpy
import pandas
import sklearn
import matplotlib
import seaborn
import pytest


def test_python_version():
    """The project should run on Python 3.12 or newer."""
    assert sys.version_info >= (3, 12)


def test_required_libraries_import():
    """Check that the initial research dependencies are available."""
    assert numpy.__version__
    assert pandas.__version__
    assert sklearn.__version__
    assert matplotlib.__version__
    assert seaborn.__version__
    assert pytest.__version__
