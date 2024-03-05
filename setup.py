from io import open

from setuptools import find_packages, setup

setup(
    name="context_similarity",
    version="0.0.1",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="context similarity text generation GenAI",
    license="MIT",
    url="https://github.com/Heyleann/context_similarity",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        "torch>=1.0.0",
        "pandas>=1.0.1",
        "transformers>=3.0.0",
        "numpy",
        "matplotlib",
        "packaging>=20.9",
        'langchain',
    ],
    include_package_data=True,
    python_requires=">=3.6",
    tests_require=["pytest"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)