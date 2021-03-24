from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='mlexpert-hub',
    version='0.0',
    description='ML Hub for 🤖 MLExpert',
    long_description_content_type="text/markdown",
    long_description=README,
    license='MIT',
    packages=find_packages(),
    author='Gokhan Egri',
    author_email='gegri@g.harvard.edu',
    keywords=['ML', "Pytorch"],
    url='https://github.com/ncthuc/elastictools',
    download_url='https://pypi.org/project/elastictools/'
)

install_requires = [
    'torch',
    'tqdm',
    'errno'
    'hashlib'
    'os'
    're'
    'shutil'
    'sys'
    'tempfile'
    'warnings'
    'zipfile'
    'urllib'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
