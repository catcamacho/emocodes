from setuptools import setup, find_packages

setup(
    name='emocodes',
    packages=find_packages(),
    version='1.0.14',
    description='A library designed to accompany the EmoCodes system.',
    author='M. Catalina Camacho',
    license='MIT',
    download_url='https://github.com/catcamacho/emocodes/archive/refs/tags/v.1.0.0.tar.gz',
    install_requires=['pandas<1.4', 'numpy', 'moviepy<2.0', 'opencv-python', 'matplotlib', 'openpyxl', 'pliers',
                      'librosa', 'seaborn', 'pingouin', 'scipy', 'scikit-learn',
                      'markdown', 'weasyprint'],
    python_requires='>=3.6'
)
