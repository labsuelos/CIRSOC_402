'''
Package setup
'''

import setuptools


setuptools.setup(
    name="cirsoc_402",
    version="0.0.1",
    author="Alejo Sfriso",
    author_email="asfriso@srk.com.ar",
    description="Pytho package for the CIRSOC 402 foundation standad project.",
    #long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/labsuelos/CIRSOC_402",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)