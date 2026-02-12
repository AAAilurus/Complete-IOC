from setuptools import setup, find_packages

package_name = 'so100_ioc_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='SO100/SO101 IOC pipeline',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'leader_log = so100_ioc_pipeline.leader_log:main',
            'ioc_fitK = so100_ioc_pipeline.ioc_fitK:main',
            'follower_run = so100_ioc_pipeline.follower_run:main',
        ],
    },
)
