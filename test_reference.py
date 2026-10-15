import socket
import tempfile
import tkinter as tk
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import ImageGrab

from main import OilPaintingApp, REFERENCE_PRESET, read_image, write_image
from local_reference import render_reference, vertical_content_bounds


def sample_portrait():
    image = np.full((180, 140, 3), (35, 28, 22), dtype=np.uint8)
    cv2.ellipse(image, (70, 75), (34, 46), 0, 0, 360, (125, 155, 190), -1)
    cv2.ellipse(image, (70, 42), (36, 22), 0, 180, 360, (20, 24, 30), -1)
    cv2.ellipse(image, (56, 70), (5, 3), 0, 0, 360, (25, 30, 36), -1)
    cv2.ellipse(image, (84, 70), (5, 3), 0, 0, 360, (25, 30, 36), -1)
    cv2.line(image, (48, 62), (62, 61), (35, 36, 40), 3)
    cv2.line(image, (78, 61), (92, 62), (35, 36, 40), 3)
    cv2.line(image, (70, 73), (65, 91), (85, 105, 130), 2)
    cv2.ellipse(image, (70, 101), (12, 5), 0, 0, 180, (55, 65, 145), 2)
    cv2.ellipse(image, (70, 165), (57, 38), 0, 180, 360, (45, 48, 78), -1)
    return image


def sample_style():
    image = np.full((220, 180, 3), (52, 45, 38), dtype=np.uint8)
    for index in range(12):
        y = 12 + index * 17
        color = (35 + index * 7, 70 + index * 5, 110 + index * 4)
        cv2.line(image, (8, y), (172, y + 18), color, 8 + index % 4)
    cv2.ellipse(image, (88, 118), (38, 65), -12, 0, 360, (90, 140, 190), -1)
    cv2.line(image, (53, 84), (76, 80), (30, 35, 45), 5)
    cv2.line(image, (98, 80), (120, 86), (30, 35, 45), 5)
    return image


class ReferenceTests(unittest.TestCase):
    def test_flat_letterbox_borders_are_excluded_from_style_transfer(self):
        padding = 80
        content = np.full((240, 180, 3), (35, 70, 110), dtype=np.uint8)
        cv2.circle(content, (90, 110), 55, (170, 130, 90), -1)
        image = np.vstack((
            np.full((padding, 180, 3), (18, 20, 24), dtype=np.uint8),
            content,
            np.full((padding, 180, 3), (15, 16, 18), dtype=np.uint8),
        ))
        self.assertEqual(vertical_content_bounds(image), (padding, padding + len(content)))
        style = sample_style()
        result = render_reference(image, style, detail=0)
        np.testing.assert_array_equal(result[:padding], image[:padding])
        np.testing.assert_array_equal(result[-padding:], image[-padding:])
        self.assertGreater(float(np.abs(result[padding:-padding].astype(float) -
                                       image[padding:-padding]).mean()), 5)

    def test_compression_stripes_in_dark_borders_are_not_detected_as_photo(self):
        padding, content_height, width = 60, 180, 180
        x = np.arange(width, dtype=np.float32)
        stripe = np.round(9 * np.sin(x * 0.16)).astype(np.uint8)
        border = np.repeat((20 + stripe)[None, :, None], padding, axis=0)
        border = np.repeat(border, 3, axis=2)
        content = np.full((content_height, width, 3), (35, 70, 110), dtype=np.uint8)
        cv2.circle(content, (90, 90), 50, (170, 130, 90), -1)
        image = np.concatenate((border, content, border), axis=0)
        self.assertGreater(float(np.std(border.astype(np.float32))), 5)
        self.assertLess(float(np.std(border.astype(np.float32))), 9)
        self.assertEqual(vertical_content_bounds(image), (padding, padding + content_height))

    def test_reference_changes_result_offline_without_color_clipping(self):
        image = sample_portrait()
        style = sample_style()
        with patch.object(socket.socket, "connect", side_effect=AssertionError("No network allowed")):
            result = render_reference(image, style, detail=0)
            other = render_reference(image, style[:, :, ::-1].copy(), detail=0)
            preserved = render_reference(image, style, detail=0, preserve_colors=True)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertGreater(float(np.abs(result.astype(float) - other.astype(float)).mean()), 5)
        self.assertLess(float(np.mean((result == 0) | (result == 255))), 0.15)
        source_luminance = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0]
        styled_luminance = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)[..., 0]
        preserved_luminance = cv2.cvtColor(preserved, cv2.COLOR_BGR2LAB)[..., 0]
        self.assertLess(abs(float(preserved_luminance.mean()) - float(source_luminance.mean())),
                        abs(float(styled_luminance.mean()) - float(source_luminance.mean())))

    def test_zero_strength_preserves_source_without_loading_model(self):
        image = np.full((50, 70, 3), 120, dtype=np.uint8)
        with patch("local_reference.load_model", side_effect=AssertionError("Model must not run")):
            np.testing.assert_array_equal(render_reference(image, image, strength=0), image)

    def test_missing_reference_is_rejected_by_reference_only_renderer(self):
        with self.assertRaisesRegex(ValueError, "referans"):
            render_reference(sample_portrait(), None, detail=0, preserve_colors=True)


class ReferenceInterfaceTests(unittest.TestCase):
    def test_reference_can_be_selected_and_converted_from_app(self):
        root = tk.Tk()
        try:
            app = OilPaintingApp(root)
            self.assertEqual(app.preset.get(), REFERENCE_PRESET)
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "portrait.png"
                style = Path(directory) / "style.png"
                write_image(source, sample_portrait())
                write_image(style, sample_style())
                with patch("main.filedialog.askopenfilename", return_value=str(source)):
                    app.open_image()
                with patch("main.filedialog.askopenfilename", return_value=str(style)):
                    app.open_reference()
                for geometry in ("880x660+0+0", "1200x780+0+0"):
                    root.geometry(geometry)
                    root.update()
                    body = app.original_preview.master.master
                    self.assertLessEqual(app.process_button.winfo_rooty() + app.process_button.winfo_height(), body.winfo_rooty() + body.winfo_height())
                with patch.object(socket.socket, "connect", side_effect=AssertionError("No network allowed")):
                    app.process_image()
                    deadline = time.monotonic() + 45
                    while app.busy and time.monotonic() < deadline:
                        root.update()
                        time.sleep(0.02)
            self.assertFalse(app.busy)
            self.assertIsNotNone(app.result)
            self.assertEqual(app.result.shape, app.original.shape)
            self.assertEqual(str(app.save_button["state"]), "normal")
            root.update()
            with tempfile.TemporaryDirectory() as directory:
                ImageGrab.grab(window=int(root.frame(), 0)).save(Path(directory) / "ui_local_reference.png")
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
