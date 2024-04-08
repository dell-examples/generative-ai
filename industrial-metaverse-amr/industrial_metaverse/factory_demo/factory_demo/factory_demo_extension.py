# Created by Scalers AI for Dell Inc.

import os

from omni.isaac.examples.base_sample import BaseSampleExtension
from omni.isaac.examples.user_examples.factory_demo import FactoryDemo


class FactoryDemoExtension(BaseSampleExtension):
    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="",
            submenu_name="",
            name="Factory Demo",
            title="Metaverse Factory Demo",
            doc_link="https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_core_hello_world.html",
            overview="This example starts a simulation of 2 nova carters in a custom factory floor.",
            file_path=os.path.abspath(__file__),
            sample=FactoryDemo(),
        )
        return
