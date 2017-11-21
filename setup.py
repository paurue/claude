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
    url='https://github.schibsted.io/pau-rue/claude',
    author=claude.__author__,
    author_email='pau.rue@schibsted.com',
    classifiers=(
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6'
    ),
    description=SHORT,
    long_description=SHORT,
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
