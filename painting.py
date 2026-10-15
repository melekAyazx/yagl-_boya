"""Layered, contour-aligned raster brush strokes rendered with OpenCV."""

import cv2
import numpy as np
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def face_detector():
    path = Path(__file__).with_name("haarcascade_frontalface_default.xml")
    # Read in memory so OpenCV also works with Turkish characters in the project path.
    storage = cv2.FileStorage(path.read_text(encoding="utf-8"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    try:
        detector = cv2.CascadeClassifier()
        if not detector.read(storage.getFirstTopLevelNode()):
            raise ValueError("Yuz bulma modeli okunamadi.")
        return detector
    finally:
        storage.release()


@lru_cache(maxsize=1)
def neural_face_detector():
    path = Path(__file__).with_name("face_detection_yunet_2023mar.onnx")
    weights = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return cv2.FaceDetectorYN.create("onnx", weights, np.empty(0, dtype=np.uint8),
                                     (320, 320), 0.75, 0.3, 5000)


def neural_face_mask(image):
    height, width = image.shape[:2]
    factor = min(1.0, 640 / max(height, width))
    small = cv2.resize(image, (max(1, round(width * factor)), max(1, round(height * factor))))
    detector = neural_face_detector()
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None:
        brightening = np.round((np.arange(256) / 255) ** 0.6 * 255).astype(np.uint8)
        _, faces = detector.detect(cv2.LUT(small, brightening))
    mask = np.zeros((height, width), dtype=np.float32)
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4] / factor
            eyes = face[4:8].reshape(2, 2)
            tilt = float(np.degrees(np.arctan2(eyes[1, 1] - eyes[0, 1], eyes[1, 0] - eyes[0, 0])))
            center = (round(x + w * 0.5), round(y + h * 0.53))
            axes = (max(1, round(w * 0.6)), max(1, round(h * 0.68)))
            cv2.ellipse(mask, center, axes, tilt, 0, 360, 1.0, -1)
    return mask


def face_detail_mask(image):
    height, width = image.shape[:2]
    mask = neural_face_mask(image)
    if np.any(mask):
        return cv2.GaussianBlur(mask, (0, 0), max(1.0, min(width, height) / 160))
    factor = min(1.0, 700 / max(height, width))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=factor, fy=factor)
    small = cv2.equalizeHist(small)
    mask = np.zeros((height, width), dtype=np.float32)
    for angle in (0, -20, 20, -40, 40):
        matrix = cv2.getRotationMatrix2D((small.shape[1] / 2, small.shape[0] / 2), angle, 1)
        rotated = cv2.warpAffine(small, matrix, (small.shape[1], small.shape[0]), borderMode=cv2.BORDER_REFLECT)
        faces = face_detector().detectMultiScale(rotated, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        inverse = cv2.invertAffineTransform(matrix)
        for x, y, w, h in faces:
            center = inverse @ np.array([x + w * 0.5, y + h * 0.53, 1]) / factor
            axes = (max(1, round(w * 0.58 / factor)), max(1, round(h * 0.68 / factor)))
            cv2.ellipse(mask, tuple(np.round(center).astype(int)), axes, angle, 0, 360, 1.0, -1)
        if len(faces):
            break
    return cv2.GaussianBlur(mask, (0, 0), max(1.0, min(width, height) / 160))


def brush_stamp(radius, angle_index, fine=False):
    extent = radius * 3 + 2
    yy, xx = np.mgrid[-extent:extent + 1, -extent:extent + 1].astype(np.float32)
    angle = angle_index * np.pi / 24
    along = xx * np.cos(angle) + yy * np.sin(angle)
    across = -xx * np.sin(angle) + yy * np.cos(angle)
    distance = (along / (radius * (1.2 if fine else 2.1))) ** 2 + (across / radius) ** 2
    # Uneven tips and longitudinal bristles give each mark a brush-like edge.
    fringe = np.sin(across * 2.3 + np.sin(across * 0.7)) * 0.055
    alpha = np.clip((1 - distance + fringe) * 5, 0, 1) * 0.92
    bristles = np.sin(across * 2.8 + np.sin(across * 1.3))
    relief = bristles * 0.018 + np.clip(-across / radius, -1, 1) * 0.014
    return alpha[..., None], relief[..., None], extent


def paint_mark(canvas, reference, y, x, stamp, rng, importance=None):
    alpha, relief, extent = stamp
    height, width = canvas.shape[:2]
    top, bottom = max(0, y - extent), min(height, y + extent + 1)
    left, right = max(0, x - extent), min(width, x + extent + 1)
    slices = (slice(top - y + extent, bottom - y + extent),
              slice(left - x + extent, right - x + extent))
    opacity = alpha[slices]
    if importance is not None:
        opacity = opacity * importance[top:bottom, left:right, None]
    color = reference[y, x] + float(rng.normal(0, 0.003 if importance is not None else 0.007))
    paint = np.clip(color + relief[slices] * (0.35 if importance is not None else 1), 0, 1)
    region = canvas[top:bottom, left:right]
    region[:] = region * (1 - opacity) + paint * opacity


def refine_face_features(canvas, source, face_mask, amount):
    """Restore thin facial contours and restrained local shading within the face."""
    reference = cv2.bilateralFilter(source, 5, 0.04, 2)
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=0.25)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=0.25)
    face_weight = float(face_mask.sum())
    brightness = float((gray * face_mask).sum() / face_weight) if face_weight > 0 else 0.35
    sensitivity = float(np.clip(0.35 / max(brightness, 0.12), 1, 2.5))
    features = np.clip(cv2.magnitude(gx, gy) * 8 * sensitivity, 0, 1)
    features = cv2.GaussianBlur(features, (0, 0), 0.45)
    weight = face_mask * amount * (0.12 + features * 0.76)
    refined = canvas * (1 - weight[..., None]) + reference * weight[..., None]
    # Small luminance relief defines eyelids, nostrils and lips without hard outlines.
    relief = np.clip(gray - cv2.GaussianBlur(gray, (0, 0), 1.1), -0.035, 0.035)
    relief *= face_mask * amount * (0.2 + features * 0.6)
    return np.clip(refined + relief[..., None], 0, 1)


def render_strokes(image, brush_size, detail):
    source = image.astype(np.float32) / 255
    height, width = source.shape[:2]
    scale = max(height, width) / 700
    radius = max(4, round((5 + brush_size * 0.13) * scale))
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    xx = cv2.GaussianBlur(gx * gx, (0, 0), 2)
    yy = cv2.GaussianBlur(gy * gy, (0, 0), 2)
    xy = cv2.GaussianBlur(gx * gy, (0, 0), 2)
    angles = (0.5 * np.arctan2(2 * xy, xx - yy) + np.pi / 2) % np.pi
    coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / (xx + yy + 1e-6)
    rng = np.random.default_rng(18)
    canvas = cv2.GaussianBlur(source, (0, 0), max(1, radius * 0.7))
    sizes = [radius, max(3, round(radius * 0.5)), max(2, round(radius * (0.3 - detail * 0.12)))]
    for layer, size in enumerate(dict.fromkeys(sizes)):
        reference = cv2.GaussianBlur(source, (0, 0), max(0.5, size * 0.32))
        error = np.max(np.abs(reference - canvas), axis=2)
        step = max(2, round(size * 1.3))
        rows, columns = np.mgrid[0:height:step, 0:width:step]
        points = np.column_stack((rows.ravel(), columns.ravel()))
        rng.shuffle(points)
        stamps = {}
        for y, x in points:
            jitter = rng.integers(-max(1, step // 2), max(2, step // 2 + 1), 2)
            y, x = int(np.clip(y + jitter[0], 0, height - 1)), int(np.clip(x + jitter[1], 0, width - 1))
            if layer and error[y, x] < (0.026 if layer == 1 else 0.04 - detail * 0.022):
                continue
            angle = float(angles[y, x])
            if coherence[y, x] < 0.35:
                angle = -0.65 + float(rng.normal(0, 0.25))
            angle_index = int(round(angle * 24 / np.pi)) % 24
            if angle_index not in stamps:
                stamps[angle_index] = brush_stamp(size, angle_index)
            paint_mark(canvas, reference, y, x, stamps[angle_index], rng)

    # Short micro-strokes recover facial features and fine high-contrast contours.
    if detail > 0:
        faces = face_detail_mask(image)
        edges = np.clip(cv2.magnitude(gx, gy) * 2, 0, 1)
        importance = np.maximum(faces, edges * coherence * 0.25) * detail
        reference = cv2.bilateralFilter(image, 5, 18, 2).astype(np.float32) / 255
        fine_size = max(1, round(scale * 0.7))
        step = max(2, fine_size)
        rows, columns = np.mgrid[0:height:step, 0:width:step]
        selected = importance[rows, columns] > 0.15
        points = np.column_stack((rows[selected], columns[selected]))
        rng.shuffle(points)
        stamps = {index: brush_stamp(fine_size, index, fine=True) for index in range(24)}
        for y, x in points:
            y, x = int(y), int(x)
            angle_index = int(round(float(angles[y, x]) * 24 / np.pi)) % 24
            paint_mark(canvas, reference, y, x, stamps[angle_index], rng, importance)

    # Retouch fine contours with softened source colors, rather than whole-photo blending.
    contours = np.clip(cv2.magnitude(gx, gy) * 1.8, 0, 0.4) * detail
    fine_colors = cv2.GaussianBlur(source, (0, 0), 0.55)
    canvas = canvas * (1 - contours[..., None]) + fine_colors * contours[..., None]
    # Match the photograph's broad illumination without restoring its texture.
    difference = cv2.cvtColor(source - canvas, cv2.COLOR_BGR2GRAY)
    lighting = cv2.GaussianBlur(difference, (0, 0), max(2, radius * 2))
    canvas = np.clip(canvas + lighting[..., None], 0, 1)
    if detail > 0:
        canvas = refine_face_features(canvas, source, faces, detail)
    return canvas
