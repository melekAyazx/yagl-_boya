import tempfile
import time
import unittest
import sys
from unittest.mock import patch
from pathlib import Path
import tkinter as tk

import cv2
import numpy as np
from PIL import ImageGrab

from main import PRESETS, REFERENCE_PRESET, OilPaintingApp, add_canvas_texture, create_oil_painting, main, parse_args, read_image, write_image
from painting import face_detail_mask, refine_face_features


class ConversionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.image = rng.integers(0, 256, (120, 180, 3), dtype=np.uint8)
        self.image[:, 90:] //= 2

    def test_zero_strength_is_identity(self):
        result = create_oil_painting(self.image, sigma_r=0, saturation=1,
                                     contrast=1, canvas_strength=0)
        self.assertLessEqual(int(np.max(np.abs(result.astype(int) - self.image.astype(int)))), 1)

    def test_detail_retains_more_of_original(self):
        options = dict(sigma_r=1, saturation=1, contrast=1, canvas_strength=0)
        low = create_oil_painting(self.image, detail=0, **options)
        high = create_oil_painting(self.image, detail=1, **options)
        self.assertLess(np.mean(np.abs(high.astype(float) - self.image)),
                        np.mean(np.abs(low.astype(float) - self.image)))
        self.assertEqual(high.shape, self.image.shape)

    def test_large_image_keeps_dimensions(self):
        image = cv2.resize(self.image, (1900, 250))
        self.assertEqual(create_oil_painting(image).shape, image.shape)
        unchanged = create_oil_painting(image, sigma_r=0, saturation=1, contrast=1, canvas_strength=0)
        self.assertLessEqual(int(np.max(np.abs(unchanged.astype(int) - image.astype(int)))), 1)

    def test_unicode_path_png_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ya\u011fl\u0131_boya.png"
            write_image(path, self.image)
            np.testing.assert_array_equal(read_image(path), self.image)

    def test_flat_skin_tone_is_not_darkened(self):
        skin = np.full((200, 250, 3), (90, 135, 185), dtype=np.uint8)
        result = create_oil_painting(skin, saturation=1, contrast=1)
        self.assertLess(abs(float(result.mean()) - float(skin.mean())), 1.5)

    def test_texture_is_subtle_without_regular_grid(self):
        flat = np.full((256, 256, 3), 140, dtype=np.uint8)
        textured = add_canvas_texture(flat, 0.16)
        self.assertLess(float(textured.std()), 2.5)
        column_means = textured[:, :, 0].mean(axis=0)
        spectrum = np.abs(np.fft.rfft(column_means - column_means.mean()))
        self.assertLess(float(spectrum.max() / max(spectrum.sum(), 1e-8)), 0.15)

    def test_default_painting_has_visible_strokes(self):
        yy, xx = np.mgrid[:180, :220]
        texture = np.sin(xx * 0.28 + yy * 0.15) * 16
        image = np.stack((80 + xx * 0.3 + texture, 70 + yy * 0.4 + texture,
                          180 - xx * 0.3 + texture), axis=2).astype(np.uint8)
        cv2.circle(image, (110, 80), 40, (90, 140, 210), -1)
        result = create_oil_painting(image)
        difference = np.abs(result.astype(float) - image.astype(float))
        self.assertGreater(float(difference.mean()), 3)
        self.assertGreater(float(np.mean(np.max(difference, axis=2) > 10)), 0.08)

    @unittest.skipUnless(Path(__file__).with_name("portrait_sample.png").exists(), "Ornek portre bulunamadi")
    def test_portrait_details_improve_without_removing_background_strokes(self):
        image = read_image(Path(__file__).with_name("portrait_sample.png"))
        mask = face_detail_mask(image)
        self.assertGreater(float(mask.sum()), 1000)
        options = dict(saturation=1, contrast=1, canvas_strength=0)
        low = create_oil_painting(image, detail=0, **options)
        high = create_oil_painting(image, detail=0.95, **options)
        low_error = np.abs(low.astype(float) - image).mean(axis=2)
        high_error = np.abs(high.astype(float) - image).mean(axis=2)
        self.assertLess(float((high_error * mask).sum()), float((low_error * mask).sum()) * 0.65)
        background = mask < 0.05
        self.assertGreater(float(high_error[background].mean()), 4)


class FaceRefinementTests(unittest.TestCase):
    def test_thin_features_are_clearer_and_background_is_unchanged(self):
        source = np.full((100, 120, 3), 0.55, dtype=np.float32)
        cv2.line(source, (30, 35), (48, 35), (0.15, 0.15, 0.15), 2)
        cv2.line(source, (70, 35), (88, 35), (0.15, 0.15, 0.15), 2)
        cv2.line(source, (60, 45), (57, 60), (0.3, 0.3, 0.3), 2)
        cv2.ellipse(source, (60, 65), (20, 8), 0, 0, 180, (0.2, 0.2, 0.2), 2)
        canvas = cv2.GaussianBlur(source, (0, 0), 2)
        mask = np.zeros(source.shape[:2], dtype=np.float32)
        mask[15:85, 15:105] = 1
        result = refine_face_features(canvas, source, mask, 0.95)
        np.testing.assert_array_equal(result[mask == 0], canvas[mask == 0])
        old_edges = cv2.Laplacian(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
        new_edges = cv2.Laplacian(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
        self.assertGreater(float(new_edges[mask > 0].var()), float(old_edges[mask > 0].var()) * 1.5)

    def test_flat_skin_does_not_gain_noise_or_change_brightness(self):
        source = np.full((80, 100, 3), (0.35, 0.5, 0.7), dtype=np.float32)
        result = refine_face_features(source, source, np.ones((80, 100), dtype=np.float32), 1)
        np.testing.assert_allclose(result, source, atol=1e-6)

    @unittest.skipUnless(Path(__file__).with_name("portrait_sample.png").exists(), "Ornek portre bulunamadi")
    def test_dark_tilted_face_detection_tracks_the_face(self):
        image = read_image(Path(__file__).with_name("portrait_sample.png"))
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 35, 1)
        rotated = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
        dark = np.clip(rotated.astype(float) * 0.4, 0, 255).astype(np.uint8)
        expected = cv2.warpAffine(face_detail_mask(image), matrix, (width, height)) > 0.5
        detected = face_detail_mask(dark) > 0.5
        overlap = np.count_nonzero(expected & detected) / max(1, np.count_nonzero(expected | detected))
        self.assertGreater(overlap, 0.7)

    @unittest.skipUnless(Path(__file__).with_name("portrait_sample.png").exists(), "Ornek portre bulunamadi")
    def test_details_are_preserved_in_dark_tilted_portrait(self):
        image = read_image(Path(__file__).with_name("portrait_sample.png"))
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 35, 1)
        image = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
        image = np.clip(image.astype(float) * 0.4, 0, 255).astype(np.uint8)
        mask = face_detail_mask(image)
        options = dict(saturation=1, contrast=1, canvas_strength=0)
        coarse = create_oil_painting(image, detail=0, **options)
        detailed = create_oil_painting(image, detail=0.95, **options)
        coarse_error = np.abs(coarse.astype(float) - image).mean(axis=2)
        detail_error = np.abs(detailed.astype(float) - image).mean(axis=2)
        self.assertLess(float((detail_error * mask).sum()), float((coarse_error * mask).sum()) * 0.65)


class ClassicalStyleTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((97, 151, 3), (50, 95, 165), dtype=np.uint8)
        cv2.circle(self.image, (70, 50), 25, (120, 175, 210), -1)
        cv2.line(self.image, (15, 80), (135, 20), (20, 45, 65), 4)

    def test_neural_style_changes_image_and_preserves_odd_dimensions(self):
        result = create_oil_painting(self.image, engine="classical", detail=0, saturation=1,
                                     contrast=1, canvas_strength=0)
        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertGreater(float(np.abs(result.astype(float) - self.image).mean()), 12)

    def test_color_preservation_keeps_palette_closer_to_original(self):
        options = dict(engine="classical", detail=0, saturation=1, contrast=1, canvas_strength=0)
        original_colors = create_oil_painting(self.image, preserve_colors=True, **options)
        style_colors = create_oil_painting(self.image, preserve_colors=False, **options)
        original_lab = cv2.cvtColor(self.image.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        preserved_lab = cv2.cvtColor(original_colors.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        styled_lab = cv2.cvtColor(style_colors.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        self.assertLess(float(np.abs(preserved_lab[..., 1:] - original_lab[..., 1:]).mean()),
                        float(np.abs(styled_lab[..., 1:] - original_lab[..., 1:]).mean()))

    def test_zero_strength_does_not_load_neural_model(self):
        with patch("main.render_classical", side_effect=AssertionError("Model should not run")):
            result = create_oil_painting(self.image, engine="classical", sigma_r=0,
                                         saturation=1, contrast=1, canvas_strength=0)
        self.assertLessEqual(int(np.max(np.abs(result.astype(int) - self.image.astype(int)))), 1)


class LocalOperationTests(unittest.TestCase):
    def test_cli_defaults_to_local_reference(self):
        with patch.object(sys, "argv", ["main.py"]):
            args = parse_args()
        self.assertEqual(args.style, "reference")
        self.assertEqual(args.brush, 115)
        self.assertEqual(args.smoothness, 1.0)
        self.assertEqual(args.detail, 0.7)
        self.assertEqual(args.saturation, 1.0)
        self.assertEqual(args.contrast, 1.0)
        self.assertEqual(args.texture, 0.11)

    def test_reference_preset_defaults_to_refined_painterly_transfer(self):
        brush, strength, detail, saturation, contrast, texture = PRESETS[REFERENCE_PRESET]
        self.assertEqual(strength, 1.0)
        self.assertEqual(detail, 0.7)
        self.assertEqual(brush, 115)
        self.assertEqual(texture, 0.11)
        self.assertEqual(saturation, 1.0)
        self.assertEqual(contrast, 1.0)

    def test_reference_preset_without_reference_falls_back_to_brush_painting(self):
        yy, xx = np.mgrid[:120, :160]
        source = np.stack((40 + xx * 0.7, 35 + yy * 0.8, 150 - xx * 0.3), axis=2).astype(np.uint8)
        options = dict(sigma_s=70, sigma_r=0.96, saturation=1, contrast=1,
                       canvas_strength=0, detail=0.8, preserve_colors=True)
        automatic = create_oil_painting(source, engine="reference", reference=None, **options)
        brush = create_oil_painting(source, engine="brush", **options)
        np.testing.assert_array_equal(automatic, brush)
        self.assertGreater(float(np.abs(automatic.astype(float) - source).mean()), 2.5)

    def test_conversion_and_save_with_network_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "original.png"
            destination = Path(directory) / "painted.png"
            image = np.full((80, 120, 3), (60, 110, 170), dtype=np.uint8)
            cv2.circle(image, (60, 40), 22, (100, 140, 190), -1)
            write_image(source, image)
            with patch("socket.socket.connect", side_effect=AssertionError("Network must not be used")), \
                 patch.dict("os.environ", {"OPENAI_API_KEY": ""}), \
                 patch.object(sys, "argv", ["main.py", str(source), str(destination)]):
                self.assertEqual(main(), 0)
            self.assertEqual(read_image(destination).shape, image.shape)


class InterfaceTests(unittest.TestCase):
    def test_background_processing_and_layout(self):
        root = tk.Tk()
        try:
            app = OilPaintingApp(root)
            self.assertEqual(app.preset.get(), REFERENCE_PRESET)
            self.assertIsNone(app.reference)
            self.assertTrue(app.preserve_colors.get())
            self.assertFalse(hasattr(app, "api_key"))
            self.assertNotIn("Yapay", " ".join(app.preset_menu["values"]))
            app.preset.set("Klasik tablo")
            app.apply_preset()
            app.original = np.full((100, 150, 3), 120, dtype=np.uint8)
            app.original_preview.set_image(app.original)
            for geometry in ("1200x780", "880x620"):
                root.geometry(geometry)
                root.update()
                for widget in (app.original_preview, app.result_preview, app.process_button, app.save_button):
                    self.assertGreater(widget.winfo_width(), 30)
                    self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(), root.winfo_rootx() + root.winfo_width())
                    self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), root.winfo_rooty() + root.winfo_height())
            app.process_image()
            self.assertTrue(app.busy)
            deadline = time.monotonic() + 15
            while app.busy and time.monotonic() < deadline:
                root.update()
                time.sleep(0.02)
            self.assertFalse(app.busy)
            self.assertIsNotNone(app.result)
            self.assertEqual(str(app.save_button["state"]), "normal")
            self.assertEqual(str(app.colors_checkbox["state"]), "normal")
            app.preset.set("Portre")
            app.apply_preset()
            self.assertEqual(str(app.colors_checkbox["state"]), "disabled")
            root.geometry("1200x780")
            root.update()
            ImageGrab.grab(window=int(root.frame(), 0)).save("ui_preview.png")
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
