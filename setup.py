from setuptools import setup, find_packages

setup(
    name='transformer_emojis',
    version='0.1',
    author='Miika Moilanen',
    author_email='mamoilanen@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    license='LICENSE',
    long_description=open('README.md').read(),
)