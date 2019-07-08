from setuptools import setup

setup(
    name="cnn-text-classify",
    version="0.0.2",
    description="CNN Text Classification",
    packages=["csmodel"],
    package_dir={"csmodel": "csmodel"},
    package_data={'csmodel': ['*.pkl']},
    author="Si Chen",
    license="MIT",
    include_package_data=True,
)