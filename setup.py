from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="SIGNAL",
    version="1.0.0",
    author="Mohd Mujtaba Akhtar, Girish, Farhan Sheth, Muskaan Singh",
    description="Graph-Augmented Instance Learning for Synthetic Speech Attribution and Open-Set Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Helixometry/SIGNAL",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
