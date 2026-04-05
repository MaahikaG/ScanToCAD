# launch/sim.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # ── Mocks (replace hardware) ──────────────────────────────────────────
        Node(package='scan_to_cad', executable='mock_sensor', output='screen'),
        Node(package='scan_to_cad', executable='mock_gantry',  output='screen'),

        # ── Real software nodes (same as full_system) ─────────────────────────
        Node(package='scan_to_cad', executable='point_cloud_pub', output='screen'),
        Node(package='scan_to_cad', executable='parsenet_trigger', output='screen'),
    ])
