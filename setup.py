import os
import re
import subprocess
import sys
from distutils import log
from pathlib import Path
from typing import List

if __name__ == "__main__":

    # Import after installing requirements, otherwise setup doesn't know versionpy_versioning
    from setuptools import setup, find_packages

    setup(
        name="br_engine",
        description="BR engine",
        version="0.1",
        packages=find_packages(),
        include_package_data=True,
        test_suite="nose2.collector.collector",
    )

# todo: fix setup.py
