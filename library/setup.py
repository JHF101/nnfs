from setuptools import setup, find_packages

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

VERSION = '0.4.0'
DESCRIPTION = 'Neural Network from Scratch'
LONG_DESCRIPTION = 'A neural network implemented from first principles with different gradient based and non-gradient based optimizers.'

# Setting up
setup(
        name="nnfs",
        version=VERSION,
        author="John-Henry Faul",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        keywords=['python', 'nnfs'],
        classifiers= [
            "Development Status :: 1 - In Progress",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: Unix",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ],
        # package_dir={"": "nnfs"},
        python_requires=">=3.6",
)