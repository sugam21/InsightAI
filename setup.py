from setuptools import setup

with open("README.md", mode="r") as f:
    long_description: str = f.readlines()

with open("LISCENSE", mode="r") as f:
    license: str = f.readlines()

setup(
    name="OptiRead",
    version="0.0.1",
    description="Image to Text Retrieval System.",
    long_description=long_description,
    long_description_content_type="markdown",
    author="Sugam Sharma, Apurva Patil",
    author_email="sugams342@gmail.com, apurvampatil17@gmail.com",
    url="https://github.com/sugam21/OptiRead",
    install_requires=["datasets", "setuptools"],
)
