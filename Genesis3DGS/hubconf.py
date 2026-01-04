import os
import json

import torch
import huggingface_hub

repo_id = "Genesis-Intelligence/internal_assets"
filename = "internal_assets/marvin_description/marvin_bimanual/urdf/tianji_bimanual_description.urdf"
repo_type = "dataset"

path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)