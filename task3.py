"""
# task3.py
This Script is designed to create a dataset for indoor-logistic scene with multi-objects.
It includes the following features:
    1. SM-Boxes
    2. Forklifts 
    3. Pallets with boxes
    4. Blue boxes
    5. RackLarge
"""

import omni.timeline
import asyncio
import omni.usd as usd
import random
import pxr.Gf as Gf
from pxr import Usd, UsdGeom
from isaacsim.sensors.camera import Camera
from omni.replicator.core import AnnotatorRegistry
import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import json

"""Task number"""
taskNum = '3'
root_dir = "E:/VScode/VSworkspace/pyworkspace/isaacsimpy"

param_time = 300
param_iter = 0
param_max_SMBoxes = 20
param_max_RackLarge = 3
param_max_BlueBoxes = param_max_RackLarge*10
param_max_Forklifts = 1
