from setuptools import setup, find_packages


description = """This is a package for doing vertebral
landmark detection from cephalogram x-ray images."""

setup(
    name="cvmt",
    version="0.0.1",
    author="Saeed Mehrang",
    author_email="saeedmehrang@gmail.com",
    description=description,
    packages=find_packages(),
)
