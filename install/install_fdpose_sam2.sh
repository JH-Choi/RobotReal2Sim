# conda create -n fdpose_sam2 python=3.10

source "$(dirname "$0")/config.sh"
FD_POSE_DIR="${PROJ_ROOT}/third_party/FoundationPose"

# Install Python dependencies from requirements_fdpose_sam2.txt
log_message "Installing Python dependencies from requirements_fdpose_sam2.txt..."
if "${PYTHON_PATH}" -m pip install --no-cache-dir -r "${CURR_DIR}/requirements_fdpose_sam2.txt"; then
    log_message "Python dependencies installed successfully."
else
    handle_error "Failed to install Python dependencies."
fi


# Install Python dependencies from requirements_fdpose.txt
log_message "Installing Python dependencies from requirements_fdpose.txt..."
if "${PYTHON_PATH}" -m pip install --no-cache-dir -r "${CURR_DIR}/requirements_fdpose.txt"; then
    log_message "Python dependencies installed successfully."
else
    handle_error "Failed to install Python dependencies."
fi

# Install NVDiffRast
log_message "Installing NVDiffRast..."
if "${PYTHON_PATH}" -m pip install --no-build-isolation --no-cache-dir "git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3" -v; then
    log_message "NVDiffRast installed successfully."
else
    handle_error "Failed to install NVDiffRast."
fi

# Install PyTorch3D
log_message "Installing PyTorch3D..."
if FORCE_CUDA=1 "${PYTHON_PATH}" -m pip install --no-build-isolation --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@stable" -v; then
    log_message "PyTorch3D installed successfully."
else
    handle_error "Failed to install PyTorch3D."
fi


# Install Python dependencies from requirements_fdpose.txt
log_message "Installing Python dependencies from requirements_fdpose.txt..."
if "${PYTHON_PATH}" -m pip install --no-cache-dir -r "${CURR_DIR}/requirements_sam2.txt"; then
    log_message "Python dependencies installed successfully."
else
    handle_error "Failed to install Python dependencies."
fi
