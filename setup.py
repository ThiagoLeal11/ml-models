from setuptools import setup, find_packages

setup(
    name='ml_models',
    description='Custom models for deeplearning.',
    version='0.0.5',
    url='https://github.com/ThiagoLeal11/ml-models',
    author='Thiago L. Pozati',
    author_email='thiagoleal11@gmail.com',
    keywords=['model', 'ml', 'machine', 'learning', 'deep', 'deeplearning'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
    ],
)
