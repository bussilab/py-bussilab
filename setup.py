from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

def version():
    from bussilab import __version__ as version
    return version

def deps():
    from bussilab import _required_ as required
    return required

def description():
    from bussilab import __doc__ as doc
    return doc.partition('\n')[0]

setup(
    name="bussilab",
    version=version(),
    author='Giovanni Bussi',
    author_email='bussi@sissa.it',
    description=description(),
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/bussilab/py-bussilab',
    packages=['bussilab'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering"
        ],
    install_requires=deps(),
    python_requires='>=3.6',
    scripts=['bin/bussilab']
)
