#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pdfparser",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Точный парсер русскоязычных PDF-документов",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pdfparser",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: Russian",
    ],
    python_requires=">=3.8",
    install_requires=[
        "dedoc>=1.0.0",
        "pdf2image>=1.16.0",
        "pdfplumber>=0.7.0",
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "img2pdf>=0.4.0",
        "PyPDF2>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "scikit-image>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pdfparser=pdfparser.cli:main",
        ],
    },
)