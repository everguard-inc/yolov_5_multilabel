from setuptools import find_packages, setup


with open("requirements.txt") as file:
    requirements = file.read().splitlines()


setup(
    name="yolov5",
    description="",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8.10,<=3.8.10",
    use_scm_version={"version_scheme": "python-simplified-semver"},
    include_package_data=True,
)