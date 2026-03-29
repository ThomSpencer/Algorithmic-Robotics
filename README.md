# Algorithmic Robotics

ROS2 lab workspace for the Succulence Rover. The main package lives under
[succulence_ws/src/succulence_rover_ros](succulence_ws/src/succulence_rover_ros) and focuses on
dead reckoning, occupancy grid mapping, and SLAM foundations.

For the full system map and week-by-week context, see
[succulence_ws/src/succulence_rover_ros/SYSTEMS_OVERVIEW.md](succulence_ws/src/succulence_rover_ros/SYSTEMS_OVERVIEW.md).

## What is in this workspace

- ROS2 package: [succulence_ws/src/succulence_rover_ros](succulence_ws/src/succulence_rover_ros)
- Config: [succulence_ws/src/succulence_rover_ros/config/params.yaml](succulence_ws/src/succulence_rover_ros/config/params.yaml)
- Launch files:
	- [succulence_ws/src/succulence_rover_ros/launch/dead_reckoning.launch.py](succulence_ws/src/succulence_rover_ros/launch/dead_reckoning.launch.py)
	- [succulence_ws/src/succulence_rover_ros/launch/slam.launch.py](succulence_ws/src/succulence_rover_ros/launch/slam.launch.py)
- Nodes (entry points in [succulence_ws/src/succulence_rover_ros/setup.py](succulence_ws/src/succulence_rover_ros/setup.py)):
	- `motion_model_node` (dead reckoning + covariance)
	- `occupancy_grid_mapper_node` (Bayesian occupancy grid)

## Build and run

From the repo root:

```bash
cd succulence_ws
colcon build --packages-select succulence_rover_ros
source install/setup.bash
```

### Dead reckoning + occupancy grid (Weeks 5-6)

```bash
ros2 launch succulence_rover_ros dead_reckoning.launch.py
```

Override frames for a physical robot if needed:

```bash
ros2 launch succulence_rover_ros dead_reckoning.launch.py \
	odom_frame:=odom base_link_frame:=base_link lidar_frame:=base_scan
```

### SLAM (Weeks 7-8)

```bash
ros2 launch succulence_rover_ros slam.launch.py
```

## Parameters and topics

All tunables live in
[succulence_ws/src/succulence_rover_ros/config/params.yaml](succulence_ws/src/succulence_rover_ros/config/params.yaml),
including:

- Topic names for simulation vs physical robot
- Frame names (`odom`, `base_link`, `map`, `lidar`)
- Motion model noise parameters (alpha values)
- Occupancy grid size, resolution, and log-odds values
- SLAM keyframe and scan matcher settings

Key topics (default simulation config):

- Inputs: `/succulence/odom`, `/succulence/scan`
- Dead reckoning outputs: `/succulence/dead_reckoning/odometry`, `/succulence/dead_reckoning/path`
- Map outputs: `/succulence/map/odom_only` (dead reckoning), `/succulence/map` (SLAM)
- SLAM outputs: `/succulence/slam/odometry`, `/succulence/slam/path`

## ROS2 package layout

```
succulence_rover_ros/
├── config/
│   └── params.yaml
├── launch/
│   ├── dead_reckoning.launch.py
│   └── slam.launch.py
├── succulence_rover_ros/
│   ├── __init__.py
│   ├── utils.py
│   ├── motion_model.py
│   └── occupancy_grid_mapper.py
├── package.xml
├── setup.py
└── setup.cfg
```

## Developer notes

- The motion model node propagates covariance using SE(2) Jacobians in
	[succulence_ws/src/succulence_rover_ros/succulence_rover_ros/utils.py](succulence_ws/src/succulence_rover_ros/succulence_rover_ros/utils.py).
- The occupancy grid mapper uses log-odds updates and Bresenham ray tracing in
	[succulence_ws/src/succulence_rover_ros/succulence_rover_ros/occupancy_grid_mapper.py](succulence_ws/src/succulence_rover_ros/succulence_rover_ros/occupancy_grid_mapper.py).
- Package metadata and dependencies are in
	[succulence_ws/src/succulence_rover_ros/package.xml](succulence_ws/src/succulence_rover_ros/package.xml).


