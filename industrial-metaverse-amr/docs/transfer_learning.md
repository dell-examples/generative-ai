# Using Replicator to generate synthetic data for transfer learning

In this document we will go over all the steps that need to be executed in order to generate synthetic data in the isaac sim environment to train AI models.

### Setting up Replicator Code.

With Isaac Sim installation there is a sample replicator code provided which can be used to understand how the replicator works.
```
/home/user/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/standalone_examples/replicator/amr_navigation.py
```

The path mentioned above is where you can find the sample code where it will generate data from the AMR's camera view. This example can be closely inspected to update and generate data for your own application.

The below snippets will help explain how that code can be updated to generate custom data.

```py
SPILL_URL = "./spill.usd"
add_reference_to_stage(usd_path=SPILL_URL, prim_path="/NavWorld/Spill")
spill = self._stage.GetPrimAtPath("/NavWorld/Spill")

def _randomize_spill_pose(self):
    min_dist_from_carter = 4
    carter_loc = self._carter_chassis.GetAttribute("xformOp:translate").Get()
    for _ in range(100):
        x, y = random.uniform(-6, 6), random.uniform(-6, 6)
        dist = (Gf.Vec2f(x, y) - Gf.Vec2f(carter_loc[0], carter_loc[1])).GetLength()
        if dist > min_dist_from_carter:
            # this is where we set the random x,y position for spill.
            spill.GetAttribute("xformOp:translate").Set((x, y, 0))
            # here we set the carter target to be the spill so it moves towards it.
            _carter_nav_target.GetAttribute("xformOp:translate").Set((x, y, 0))
            break

def _setup_sdg(self):
    # Disable capture on play and async rendering
    carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
    carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)
    carb.settings.get_settings().set("/app/asyncRendering", False)

    # this is where you define the writer that is used to record the data.
    writer = rep.WriterRegistry.get("KittiWriter")
    writer.initialize(output_dir=_out_dir)

    if not _use_temp_rp:
        _setup_render_products()

def _setup_render_products(self):
    print(f"[NavSDGDemo] Creating render products")
    rp_left = rep.create.render_product(
        "/NavWorld/CarterNav/chassis_link/stereo_cam_left/stereo_cam_left_sensor_frame/camera_sensor_left",
        (512, 512),
        name="left_sensor",
        force_new=True,
    )
    rp_right = rep.create.render_product(
        "/NavWorld/CarterNav/chassis_link/stereo_cam_right/stereo_cam_right_sensor_frame/camera_sensor_right",
        (512, 512),
        name="right_sensor",
        force_new=True,
    )
    _render_products = [rp_left, rp_right]

    # here you attach the sensors to capture the data.
    writer.attach(_render_products)
```

The above mentioned are some of the snippets from the actual replicator code where you can make edits according to your custom requirements. In place of spill any object can be loaded to create the dataset on. For the writers available refer to the [Omniverse Writers](https://docs.omniverse.nvidia.com/py/replicator/1.10.10/source/extensions/omni.replicator.core/docs/API.html#writers) documentation.

To understand more details refer to the [Official replicator](https://docs.omniverse.nvidia.com/isaacsim/latest/replicator_tutorials/tutorial_replicator_amr_navigation.html) documentation and understand how the tutorial works.

Once the code has run for N number of iterations the data generated will be stored in the output_directory set on the code.
This data can be then used to annotate and create a dataset to train the AI models.

### Curating the Synthetic Dataset

For creating the synthetic dataset for training the chemical spill segmentation model, the below variations were introduced the factory demo environment using NVIDIA® Omniverse™ Replicator.

* Types of chemical spill
* Location of the chemical spill on the factory floor
* Lighting of the factory floor
* Environment of the factory

Based on the objects to be detected/segmented, the variation introduced on the simulated environment needs to updated.

### Fine Tuning Model

Once the required dataset is created, the synthetic dataset combined with real world dataset can be used to train or improve upon an existing computer vision model.

The NVIDIA® Omniverse™ Replicator supports multiple dataset formats including KITTI. The KITTI formatted dataset can be leveraged for training a segmentation model.

Based on the support for the training libraries/platform, there might be a need to annotate the dataset manually. In those cases, tools like [CVAT](https://github.com/opencv/cvat) can be leveraged.

With the annotated dataset, the chemical spill was trained on an [YOLOv8s](https://github.com/ultralytics/ultralytics) model. The dataset was converted to COCO format using the CVAT annotation tool for training. For more information on training a YOLOv8 model, refere [Train](https://docs.ultralytics.com/modes/train/) section in ultralytics official documentation.

The model utilized for this solution was trained with 300 images with chemical spill as primary detection class. The training resulted in a model with ~95% accuracy.
