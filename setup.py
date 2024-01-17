from distutils.command.build import build as _build

import setuptools


class build(_build):  # pylint: disable=invalid-name
    """A build command class that will be invoked during package install.

    The package built using the current setup.py will be staged and later
    installed in the worker using `pip install package'. This class will be
    instantiated during install for this specific scenario and will trigger
    running the custom commands specified.
    """


setuptools.setup(
    name="diffusion_pytorch",
    version="0.0.1",
    description=("Tool for training a diffusion model."),
    author="m6n3",
    license="MIT",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers for the list
    # of values.
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    test_suite="nose2.collector.collector",
    packages=setuptools.find_packages(),
    package_data={"diffusion_pytorch": ["diffusion_pytorch/testdata/*"]},
    cmdclass={
        # Command class instantiated and run during pip install scenarios.
        "build": build,
    },
)
