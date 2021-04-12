from setuptools import setup

setup(
    name='emocodes',
    packages=['emocodes'],
    version='0.1.1',
    description='A library designed to accompany the EmoCodes system.',
    author='M. Catalina Camacho',
    license='MIT',
    download_url='https://github.com/catcamacho/emocodes/',
    install_requires=['pandas', 'numpy', 'moviepy', 'opencv-python', 'matplotlib', 'openpyxl','sphinx-material'],
    python_requires='>=3.6'
)
