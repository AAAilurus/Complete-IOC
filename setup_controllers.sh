#!/bin/bash

echo "Setting up controllers for SO100 and SO101..."

# SO100
ros2 param set /so100/controller_manager joint_state_broadcaster.type joint_state_broadcaster/JointStateBroadcaster
ros2 param set /so100/controller_manager arm_position_controller.type position_controllers/JointGroupPositionController

ros2 control load_controller -c /so100/controller_manager joint_state_broadcaster
ros2 control load_controller -c /so100/controller_manager arm_position_controller

ros2 param set /so100/arm_position_controller joints "['Shoulder_Pitch','Elbow']"
ros2 param set /so100/arm_position_controller interface_name position

ros2 control set_controller_state -c /so100/controller_manager joint_state_broadcaster inactive
ros2 control set_controller_state -c /so100/controller_manager joint_state_broadcaster active
ros2 control set_controller_state -c /so100/controller_manager arm_position_controller inactive
ros2 control set_controller_state -c /so100/controller_manager arm_position_controller active

# SO101
ros2 param set /so101/controller_manager joint_state_broadcaster.type joint_state_broadcaster/JointStateBroadcaster
ros2 param set /so101/controller_manager arm_position_controller.type position_controllers/JointGroupPositionController

ros2 control load_controller -c /so101/controller_manager joint_state_broadcaster
ros2 control load_controller -c /so101/controller_manager arm_position_controller

ros2 param set /so101/arm_position_controller joints "['Shoulder_Pitch','Elbow']"
ros2 param set /so101/arm_position_controller interface_name position

ros2 control set_controller_state -c /so101/controller_manager joint_state_broadcaster inactive
ros2 control set_controller_state -c /so101/controller_manager joint_state_broadcaster active
ros2 control set_controller_state -c /so101/controller_manager arm_position_controller inactive
ros2 control set_controller_state -c /so101/controller_manager arm_position_controller active

echo ""
echo "=== SO100 Controllers ==="
ros2 control list_controllers -c /so100/controller_manager

echo ""
echo "=== SO101 Controllers ==="
ros2 control list_controllers -c /so101/controller_manager

echo ""
echo "✓ Controllers ready!"
