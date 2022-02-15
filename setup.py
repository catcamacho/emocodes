from setuptools import setup, find_packages

setup(
    name='emocodes',
    packages=find_packages(),
    version='1.0.13',
    description='A library designed to accompany the EmoCodes system.',
    author='M. Catalina Camacho',
    license='MIT',
    download_url='https://github.com/catcamacho/emocodes/archive/refs/tags/v.1.0.0.tar.gz',
    install_requires=['pandas', 'numpy', 'moviepy', 'opencv-python', 'matplotlib', 'openpyxl', 'pliers',
                      'librosa', 'python-magic-bin==0.4.14', 'seaborn', 'pingouin', 'scipy', 'scikit-learn',
                      'markdown', 'weasyprint'],
    python_requires='>=3.6'
)
