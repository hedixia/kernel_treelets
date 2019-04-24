import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="kernel_treelets",
	version="0.1.0",
	author="Hedi Xia",
	author_email="xiahedi@gmail.com",
	description="Kernel Treelet",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/hedixia/kernel_treelets",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
		"Operating System :: OS Independent",
	],
)
