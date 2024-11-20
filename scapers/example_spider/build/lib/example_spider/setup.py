from setuptools import setup, find_packages

setup(
    name='example_spider',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'scrapy',
    ],
    entry_points={
        'scrapy': [
            'spiders = example_spider.spiders',
        ],
    },
)
