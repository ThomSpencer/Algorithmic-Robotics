#!/bin/bash
echo "source /opt/ros/jazzy/local_setup.bash" >> ~/.bashrc
echo "source /workspace/succulence_ws/install/local_setup.bash" >> ~/.bashrc

exec /bin/bash