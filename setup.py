from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

def version():
    from bussilab import __version__
    return __version__

def deps():
    from bussilab import _required_
    return _required_

def description():
    import ast
    M = ast.parse(''.join(open("bussilab/__init__.py")))
    d = ast.get_docstring(M)
    if d is None:
        return ""
    # grab first line
    return d.partition('\n')[0]

setup(
    name="bussilab",
    version=version(),
    description=description(),
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Giovanni Bussi',
    author_email='bussi@sissa.it',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering"
        ],
    packages=['bussilab'],
    install_requires=deps(),
    python_requires='>=3.6',
    scripts=['bin/bussilab']
)
