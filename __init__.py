import os
import sys
import folder_paths  # ComfyUI
# Proceed with node setup
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

custom_nodes_path = os.path.join(folder_paths.base_path, "custom_nodes")
my_path = os.path.join(custom_nodes_path, "ComfyUI-facefusion")
sys.path.append(my_path)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
