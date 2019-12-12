from setuptools import setup

setup(
    name='char_rbm',
    version='0.1.2',
    packages=['char_rbm'],
    url='https://github.com/OpenJarbas/simple_char_rbm',
    license='apache2.0',
    install_requires=["scikit-learn", "colorama"],
    author='jarbasai',
    author_email='jarbasai@mailfence.com',
    description='simple interface to train and sample char-rbms'
)
