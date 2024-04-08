# Isaac Sim Advanced User Guide

Refer [Isaac Sim API documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/reference_python_api.html) for more in depth use of all the functionality available with the isaac sim and omniverse. Below are some of the configurations used by us while developing the factory demo application.

## Table of Contents
* [USD file configurations](#usd-file-configurations)
* [Camera configurations](#camera-configurations)
* [Configuring a robot dog in the simulation](#configuring-a-robot-dog-in-the-simulation)

## USD file configurations

For loading .usd files in our application we have used the create_prim method from prims utils.
```py
from omni.isaac.core.utils.prims import create_prim

self.factory = create_prim(
    prim_path="/World/factory",
    prim_type="Xform",
    position=np.array([0.0, 0.0, 0.0]),
    usd_path="./factory_demo.usd"
)
```
The above is a sample use to load a custom .usd file and below you can find extra arguments that can be used while using the create_prim method.

| Variable | Type | Description |
|---- | ------ | --- |
| `prim_path`| str | The path of the new prim. |
| `prim_type` | str | Prim type name |
| `position`  | Sequence[float], optional | prim position (applied last) |
| `translation` | Sequence[float], optional | prim translation (applied last) |
| `orientation` | Sequence[float], optional | prim rotation as quaternion |
| `scale`  | np.ndarray(3), optional | scaling factor in x, y, z. |
| `usd_path` | str, optional | Path to the USD that this prim will reference. |
| `semantic_label` | str, optional | Semantic label. |
| `semantic_type` | str, optional | set to “class” unless otherwise specified. |
| `attributes` | dict, optional | Key-value pairs of prim attributes to set. |

## Camera configurations

For creating a camera in the simulation to capture live video from the robots we have made use of the **Camera** class from the sensors in isaac sim.

```py
from omni.isaac.sensor.scripts.camera import Camera

world.scene.add(Camera(prim_path="/World/Carter1/chassis_link/AMRCam1", name="AMRCam1", frequency=20, resolution=(1920, 1080), translation=[0,0,1], orientation=None))
```
The above is a sample example of how to attach a camera on a robot. In the prim path we have selected **chassis_link** as this inside the robot which will move in the simulation and we want our camera to move with the robot. Similarly a camera can be placed anywhere in the world in a static position as well.

Below are some of the extra arguments that can be used to configure camera.

| Variable | Type | Description |
|---- | ------ | --- |
| `prim_path` | str | prim path of the Camera Prim to encapsulate or create. |
| `name` | str, optional | shortname to be used as a key by Scene class. |
| `frequency` | int, optional | Frequency of the sensor (i.e: how often is the data frame updated). Defaults to None. |
| `dt` | [str], optional | dt of the sensor (i.e: period at which a the data frame updated). Defaults to None. |
| `resolution` | Tuple[int, int], optional | resolution of the camera (width, height). Defaults to None. |
| `position` | Sequence[float], optional | position in the world frame of the prim. shape is (3, ). Defaults to None, which means left unchanged. |
| `translation` | Sequence[float], optional | translation in the local frame of the prim (with respect to its parent prim). shape is (3, ). Defaults to None, which means left unchanged. |
| `orientation` | Sequence[float], optional | quaternion orientation in the world/ local frame of the prim (depends if translation or position is specified). quaternion is scalar-first (w, x, y, z). shape is (4, ). Defaults to None, which means left unchanged. |
| `render_product_path` | str | path to an existing render product, will be used instead of creating a new render product the resolution and camera attached to this render product will be set based on the input arguments. |

> Note: Using same render product path on two Camera objects with different camera prims, resolutions is not supported Defaults to None

For accessing the frames captured by the camera and some extra methods that can be used with the camera class refer to the [Isaac Sim Camera](https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.sensor/docs/index.html?highlight=camera#module-omni.isaac.sensor.scripts.camera) documentation.

The method used in our factory demo application is provided below.
```py
frame = camera.get_current_frame()
image_data = frame['rgba']
```

## Configuring a robot dog in the simulation

In this example we have used the **Unitree A1** quadruped robot to load in the simulation and make it move around using waypoints.

```py
from omni.isaac.quadruped.robots import Unitree

a1 = world.scene.add(
    Unitree(
        prim_path="/World/A1",
        name="A1",
        position=np.array([0, 0, 0.40]),
        physics_dt=physics_dt,
        model="A1",
        way_points=way_points,
    )
)

a1.set_state(a1._default_a1_state)

# Below commands to be run in physics step callback.
a1._qp_controller.switch_mode()
a1.advance(step_size, base_command, path_follow)
```

After importing the a1 robot in the simulation we need to use the controllers linked to Unitree quadrupeds to move the robot.

The steps after importing like setting the default state of the robot, switching mode from manual control to automatic control and using advance call to make the robot move are some of the functions we use to configure the robot states and movements while the simulation is running.
Refer more on the [Isaac Sim API](https://docs.omniverse.nvidia.com/isaacsim/latest/reference_python_api.html) documentation and search for Unitree to get more details on advanced usage of the quadruped robots.
