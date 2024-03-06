from io import open

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="context-similarity",
    version="0.0.1",
    author="Lihan Zhou",
    author_email="heyleann@outlook.com",
    description="A package that enables comparison between two context using bge.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Heyleann/context_similarity",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)