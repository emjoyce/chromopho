from setuptools import setup, find_packages


setup(
    name = "chromopho",
    version = "0.0.2",
    author_email = "emjoyce@ucsb.edu",
    description = "a simulation of bipolar subtypes as color filters",
    url = "https://github.com/emjoyce/chromopho.git",
    packages = find_packages(),
    install_requires = [
        line.strip() for line in open("requirements.txt", "r") if line.strip() and not line.startswith("#")
        ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = ">=3.6",
)

