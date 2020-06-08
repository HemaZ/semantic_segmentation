from setuptools import setup

package_name = 'semantic_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ("lib/python3.6/site-packages/semantic_segmentation/enet-cityscapes",
         ["semantic_segmentation/enet-cityscapes/enet-classes.txt", "semantic_segmentation/enet-cityscapes/enet-colors.txt", "semantic_segmentation/enet-cityscapes/enet-model.net"])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hema',
    maintainer_email='ibrahim.essam1995@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segment = semantic_segmentation.semantic_segmentation:main',
        ],
    },
)
