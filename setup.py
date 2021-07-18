from setuptools import setup, find_packages

setup(
    name='emocodes',
    packages=find_packages(),
    version='0.1.1',
    description='A library designed to accompany the EmoCodes system.',
    author='M. Catalina Camacho',
    license='MIT',
    download_url='https://github.com/catcamacho/emocodes/',
    install_requires=['spleeter', 'pandas', 'numpy', 'moviepy', 'opencv-python', 'matplotlib', 'openpyxl', 'pliers', 'librosa',
                      'python-magic-bin==0.4.14', 'seaborn', 'pingouin','scipy'],
    python_requires='>=3.6'
)
