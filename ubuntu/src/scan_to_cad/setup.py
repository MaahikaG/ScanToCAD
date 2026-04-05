from setuptools import setup
import os
from glob import glob

setup(
    name='scan_to_cad',
    packages=['scan_to_cad', 'scan_to_cad.mock'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/scan_to_cad']),
        ('share/scan_to_cad', ['package.xml']),
        (os.path.join('share', 'scan_to_cad', 'launch'), glob('launch/*.py')),  # ← this
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            # Real hardware nodes
            'motor_controller    = scan_to_cad.motor_controller:main',
            'odom_tf_pubs        = scan_to_cad.odom_tf_pubs:main',
            'pan_tilt_controller = scan_to_cad.pan_tilt_controller:main',
            'sensor_pub          = scan_to_cad.sensor_pub:main',
            'point_cloud_pub     = scan_to_cad.point_cloud_pub:main',
            'lcd_display         = scan_to_cad.lcd_display:main',

            # Mock nodes (for Docker sim testing)
            'mock_sensor        = scan_to_cad.mock.mock_sensor:main',
            'mock_gantry         = scan_to_cad.mock.mock_gantry:main',

            'parsenet_trigger = scan_to_cad.parsenet_trigger:main',
        ],
    },

)