import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pose_pipeline",
    version="0.1.0",
    author="James Cotton",
    author_email="peabody124@gmail.com",
    description="Video pose analysis pipelines for DataJoint.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/peabody124/PosePipeline",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
