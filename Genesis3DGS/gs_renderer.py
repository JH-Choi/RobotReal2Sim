from pathlib import Path
import numpy as np
import torch
import copy
import random
import time
import open3d as o3d
import math
import transforms3d
import kornia
import cv2
from sklearn.neighbors import NearestNeighbors

from transform_utils import setup_camera, Rt_to_w2c, interpolate_motions
from sh_utils import C0
from gs_processor import GSProcessor

from diff_gaussian_rasterization import GaussianRasterizer

class GSRenderer:
    def __init__(self, local_rank=0):

        self.device = f'cuda:{local_rank}'

        self.rendervar = {}
        self.rendervar_full = {}

        self.sp = GSProcessor()
    
    def get_state(self):