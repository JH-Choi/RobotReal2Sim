# import sys
# sys.path.append("../Genesis")
# from tests.utils import get_hf_dataset
# from huggingface_hub import snapshot_download


# local_dir = 'tmp_data'

# repo_name = "internal_assets"
# pattern="marvin_description/marvin_bimanual/urdf/tianji_bimanual_description.urdf"
# # Try downloading the assets
# asset_path = snapshot_download(
#     repo_type="dataset",
#     repo_id=f"Genesis-Intelligence/{repo_name}",
#     allow_patterns=pattern,
#     max_workers=1,
#     local_dir=local_dir,
# )


import os
import sys
sys.path.append("../Genesis")

from huggingface_hub import snapshot_download

local_dir = "tmp_data"

# 1) Choose the correct dataset id.
# If the assets are actually under `assets`, change this to:
# repo_id = "Genesis-Intelligence/assets"
repo_id = "Genesis-Intelligence/internal_assets"

# 2) Pattern for the URDF you want
pattern = "marvin_description/marvin_bimanual/urdf/tianji_bimanual_description.urdf"

# 3) Get token from env (or hardcode for quick testing)
hf_token = os.environ.get("HF_TOKEN")  # set this in your shell before running
print('hf_token:', hf_token)

asset_path = snapshot_download(
    repo_type="dataset",
    repo_id=repo_id,
    allow_patterns=[pattern],  # list is safer than bare string
    max_workers=1,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    token=hf_token,            # this is the key part for private repos
)

print("Snapshot downloaded to:", asset_path)
urdf_path = os.path.join(
    asset_path,
    "marvin_description",
    "marvin_bimanual",
    "urdf",
    "tianji_bimanual_description.urdf",
)
print("URDF path:", urdf_path)
