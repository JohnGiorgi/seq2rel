# -*- coding: utf-8 -*-

# DO NOT EDIT THIS FILE!
# This file has been autogenerated by dephell <3
# https://github.com/dephell/dephell

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os.path

readme = ""
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, "README.rst")
if os.path.exists(readme_path):
    with open(readme_path, "rb") as stream:
        readme = stream.read().decode("utf8")

setup(
    long_description=readme,
    name="seq2rel",
    version="0.1.0",
    description="A Python package that makes it easy to use sequence-to-sequence (seq2seq) learning for information extraction.",
    python_requires="==3.*,>=3.7.1",
    project_urls={
        "homepage": "https://github.com/johngiorgi/seq2rel",
        "repository": "https://github.com/johngiorgi/seq2rel",
    },
    author="johngiorgi",
    author_email="johnmgiorgi@gmail.com",
    license="Apache-2.0",
    keywords="named entity recognition relation extraction information extraction seq2seq",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    packages=[
        "seq2rel",
        "seq2rel.common",
        "seq2rel.data",
        "seq2rel.data.dataset_readers",
        "seq2rel.metrics",
        "seq2rel.models",
        "seq2rel.modules",
        "seq2rel.modules.attention",
    ],
    package_dir={"": "."},
    package_data={},
    install_requires=[
        "allennlp==2.*,>=2.4.0",
        "allennlp-models==2.*,>=2.4.0",
        "more-itertools==8.*,>=8.7.0",
        "typer[all]==0.*,>=0.3.2",
        "validators==0.*,>=0.18.2",
    ],
    extras_require={
        "dev": [
            "black==21.*,>=21.4.0.b2",
            "codecov==2.*,>=2.1.10",
            "coverage==5.*,>=5.5.0",
            "dephell[full]==0.*,>=0.8.3",
            "flake8==3.*,>=3.9.1",
            "hypothesis==6.*,>=6.10.0",
            "mypy==0.*,>=0.812.0",
            "pytest==6.*,>=6.2.3",
            "pytest-cov==2.*,>=2.11.1",
        ],
        "optuna": ["allennlp-optuna==0.*,>=0.1.5"],
    },
)
