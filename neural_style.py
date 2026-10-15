"""Local Starry Night style transfer using Johnson et al.'s pretrained model."""

from functools import lru_cache
from pathlib import Path
import shutil
import tempfile
import threading

import cv2
import numpy as np

from painting import face_detail_mask


MODEL_PATH = Path(__file__).with_name("starry_night.t7")
MODEL_LOCK = threading.Lock()
MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)


@lru_cache(maxsize=1)
def style_network():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Klasik tablo modeli bulunamadi: starry_night.t7")
    try:
        network = cv2.dnn.readNetFromTorch(str(MODEL_PATH))
    except cv2.error:
        # Older Windows OpenCV builds cannot open non-ASCII model paths.
        with tempfile.TemporaryDirectory(prefix="oil_style_") as directory:
            temporary = Path(directory) / "style.t7"
            shutil.copyfile(MODEL_PATH, temporary)
            network = cv2.dnn.readNetFromTorch(str(temporary))
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return network


def render_classical(image, brush_size=85, detail=0.6, preserve_colors=True):
    height, width = image.shape[:2]
    # Lower inference resolution produces larger learned paint marks.
    limit = int(np.clip(512 * 85 / max(10, brush_size), 320, 768))
    factor = min(1.0, limit / max(height, width))
    work = cv2.resize(image, (max(1, round(width * factor)), max(1, round(height * factor))), interpolation=cv2.INTER_AREA)
    h, w = work.shape[:2]
    padded = cv2.copyMakeBorder(work, 0, max(32, (h + 3) // 4 * 4) - h,
                               0, max(32, (w + 3) // 4 * 4) - w, cv2.BORDER_REFLECT)
    blob = cv2.dnn.blobFromImage(padded, 1, (padded.shape[1], padded.shape[0]),
                                tuple(float(v) for v in MEAN), swapRB=False, crop=False)
    with MODEL_LOCK:
        network = style_network()
        network.setInput(blob)
        output = network.forward()[0].transpose(1, 2, 0) + MEAN
    painted = np.clip(output[:h, :w], 0, 255).astype(np.uint8)
    painted = cv2.medianBlur(painted, 3).astype(np.float32) / 255
    original = work.astype(np.float32) / 255
    if preserve_colors:
        source_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        style_lab = cv2.cvtColor(painted, cv2.COLOR_BGR2LAB)
        style_lab[..., 1:] = source_lab[..., 1:] * 0.8 + style_lab[..., 1:] * 0.2
        illumination = cv2.GaussianBlur(source_lab[..., 0] - style_lab[..., 0], (0, 0), 10)
        style_lab[..., 0] = np.clip(style_lab[..., 0] + illumination * 0.65, 0, 100)
        painted = np.clip(cv2.cvtColor(style_lab, cv2.COLOR_LAB2BGR), 0, 1)
    if detail > 0:
        faces = face_detail_mask(work)
        reference = cv2.bilateralFilter(original, 5, 0.04, 2)
        if not preserve_colors:
            ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
            ref_lab[..., 1:] = cv2.cvtColor(painted, cv2.COLOR_BGR2LAB)[..., 1:]
            reference = np.clip(cv2.cvtColor(ref_lab, cv2.COLOR_LAB2BGR), 0, 1)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, scale=0.25)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, scale=0.25)
        features = np.clip(cv2.magnitude(gx, gy) * 8, 0, 1)
        opacity = faces * detail * (0.08 + features * 0.32)
        painted = painted * (1 - opacity[..., None]) + reference * opacity[..., None]
    return cv2.resize(painted, (width, height), interpolation=cv2.INTER_CUBIC)
