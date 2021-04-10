from setuptools import setup

setup(
        name='feyndea',
        version='1.0',
        py_modules=['model'],
        install_requires=['pandas', 'pulp', 'click'],
        entry_points='''
        [console_scritps]
        dea=model:dea
        '''

)
