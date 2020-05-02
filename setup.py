from setuptools import setup

setup(
    name="nnhealpix",
    version="0.2.1",
    description="",
    url="",
    author="Nicoletta Krachmalnicoff, Maurizio Tomasi",
    author_email="nkrach@sissa.it, maurizio.tomasi@unimi.it",
    license="MIT",
    packages=["nnhealpix", "nnhealpix.layers"],
    package_dir={"nnhealpix": "nnhealpix", "nnhealpix.layers": "nnhealpix/layers"},
    package_data={"nnhealpix": ["ancillary_files/*"]},
    include_package_data=True,
    zip_safe=False,
)
