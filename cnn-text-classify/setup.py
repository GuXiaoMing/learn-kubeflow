from setuptools import setup

setup(
    name="cnn-text-classify",
    version="0.0.2",
    description="CNN Text Classification",
    packages=["csmodel"],
    package_data={'csmodel': ['csmodel/*.pkl']},
    include_package_data=True,
    author="Si Chen",
    license="MIT",
    include_package_data=True,
)