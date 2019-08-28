import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

def readme():
    with open('README.rst') as f:
        return f.read()

setuptools.setup(
    name='Supercell',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/tnecio/supercell_core',
    license='GPLv3',
    author='Tomasz Necio',
    author_email='Tomasz.Necio@fuw.edu.pl',
    description='Package for investigation of mulitlayer 2D heterostructures lattices',
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    include_package_data=True
)
