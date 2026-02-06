from setuptools import find_packages, setup

package_name = 'so100_lqr'

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
    description='LQR outer-loop commander for SO-100 2DOF',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lqr_outer_loop = so100_lqr.lqr_outer_loop:main',
        ],
    },
)
