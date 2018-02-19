try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import claude

SHORT = 'claude implements Information theoretic measures'

setup(
    name='claude',
    version=claude.__version__,
    packages=[
        'claude',
    ],
    url='https://github.com/paurue/claude',
    author=claude.__author__,
    author_email='pau.rue@gmail.com',
    classifiers=(
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6'
    ),
    description=SHORT,
    long_description=SHORT,
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
