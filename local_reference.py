"""CPU-only reference-guided style transfer; no API or runtime downloads."""

from functools import lru_cache
from pathlib import Path
import threading

import cv2
import numpy as np


MODEL_DIR = Path(__file__).with_name("vendor_microast")
LOCK = threading.Lock()


@lru_cache(maxsize=1)
def load_model():
    import torch
    from vendor_microast.net_microAST import Encoder, Modulator, Decoder, TestNet

    torch.set_num_threads(min(4, torch.get_num_threads()))
    modules = (Encoder(), Encoder(), Modulator(), Decoder())
    names = ("content_encoder", "style_encoder", "modulator", "decoder")
    for module, name in zip(modules, names):
        path = MODEL_DIR / f"{name}_iter_160000.pth.tar"
        module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        module.eval()
    return TestNet(*modules).eval()


def image_tensor(image, limit):
    import torch

    height, width = image.shape[:2]
    factor = min(1.0, limit / max(height, width))
    resized = cv2.resize(image, (max(1, round(width * factor)), max(1, round(height * factor))), interpolation=cv2.INTER_AREA)
    h, w = resized.shape[:2]
    padded = cv2.copyMakeBorder(resized, 0, max(16, (h + 3) // 4 * 4) - h,
                               0, max(16, (w + 3) // 4 * 4) - w, cv2.BORDER_REFLECT)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0), (h, w)


def vertical_content_bounds(image):
    height = image.shape[0]
    row_variation = np.std(image.astype(np.float32), axis=(1, 2))
    threshold = max(9.0, float(np.percentile(row_variation, 90)) * 0.25)
    active_rows = np.flatnonzero(row_variation >= threshold)
    if active_rows.size == 0:
        return 0, height
    top, bottom = int(active_rows[0]), int(active_rows[-1]) + 1
    minimum_border = max(12, round(height * 0.04))
    if (top >= minimum_border and height - bottom >= minimum_border
            and bottom - top >= height * 0.5):
        return top, bottom
    return 0, height


def render_reference(image, reference, strength=0.85, detail=0.7, preserve_colors=False, style_size=512):
    if reference is None:
        raise ValueError("Bir referans gorseli olmadan stil aktarimi yapilamaz.")
    strength = float(np.clip(strength, 0, 1))
    if strength == 0:
        return image.copy()
    import torch
    from painting import face_detail_mask, refine_face_features

    height, width = image.shape[:2]
    top, bottom = vertical_content_bounds(image)
    content_image = image[top:bottom]
    content_height, content_width = content_image.shape[:2]
    content, (h, w) = image_tensor(content_image, 768)
    style, _ = image_tensor(reference, int(np.clip(style_size, 256, 768)))
    with LOCK, torch.inference_mode():
        # Released weights are stable at full feature modulation; blend intensity in image space.
        output = load_model()(content, style, 1.0)[0].clamp(0, 1)
    rgb = output.permute(1, 2, 0).numpy()[:h, :w]
    painted = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    original = cv2.resize(content_image, (w, h), interpolation=cv2.INTER_AREA)
    original_float = original.astype(np.float32) / 255
    painted = painted * strength + original_float * (1 - strength)
    if preserve_colors:
        lab = cv2.cvtColor(painted, cv2.COLOR_BGR2LAB)
        source_lab = cv2.cvtColor(original_float, cv2.COLOR_BGR2LAB)
        lab[..., 0] = source_lab[..., 0] * 0.7 + lab[..., 0] * 0.3
        lab[..., 1:] = source_lab[..., 1:] * 0.85 + lab[..., 1:] * 0.15
        painted = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0, 1)
    if detail > 0:
        painted = refine_face_features(painted, original_float, face_detail_mask(original), float(np.clip(detail, 0, 1)) * 0.45)
    painted = cv2.resize(painted, (content_width, content_height), interpolation=cv2.INTER_CUBIC)
    result = image.copy()
    result[top:bottom] = np.clip(painted * 255, 0, 255).astype(np.uint8)
    return result
