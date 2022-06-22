"""
Package installation module
"""

from setuptools import setup, find_packages


def readme():
    """
    Reads README.md file to extract detailed information from the
    project. The file has to be in the main folder of the package
    """
    with open("README.md") as file_stream:
        return file_stream.read()

setup(
    name="face_detection_mtcnn",
    version="0.0.1",
    description="Face detection using MTCNN architecture",
    long_description=readme(),
    classifiers=[
        "License :: OSI Approved :: MIT Licence",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],

    keywords="",
    url="",
    author="Fernando Herrera",
    author_email="fernando.j.herrera@gmail.com",
    license="MIT",

    python_requires=">=3.7",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pytest==5.1.1",
        "pytest-cov==2.9.0",
        "numpy==1.22.0",
        "torch==1.5.0",
        "torchvision==0.6.0",
    ],

    include_package_data=True,
    zip_safe=False,

    entry_points={
        #"console_scripts": ["start-worker=worker_main:main"]
    }
)
