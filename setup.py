#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'seqtag_keras'
DESCRIPTION = 'Easy to use BiLSTM+CRF sequence tagging for text.'
URL = 'https://github.com/bedapudi6788/seqtag-keras'
EMAIL = 'praneethbedapudi@gmail.com'
AUTHOR = 'BEDAPUDI PRANEETH'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = '1.1.0'


def get_requires(filename: str):
    requirements = []
    filepath = Path(__file__).parent / filename
    with filepath.open("rt") as req_file:
        for line in req_file.read().splitlines():
            line = line.strip()
            if not line.startswith("#") and len(line):
                requirements.append(line)
    return requirements


project_requirements = get_requires("requirements.txt")


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=project_requirements,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
