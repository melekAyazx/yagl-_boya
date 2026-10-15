import argparse
import queue
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageTk
from painting import render_strokes
from neural_style import render_classical
from local_reference import render_reference


APP_TITLE = "Ya\u011fl\u0131 Boya At\u00f6lyesi 8.0"
REFERENCE_PRESET = "Referansl\u0131 ya\u011fl\u0131 boya"
PRESETS = {
    REFERENCE_PRESET: (70, 0.96, 0.8, 1.0, 1.0, 0.02),
    "Klasik tablo": (85, 0.95, 0.6, 1.04, 1.0, 0.035),
    "Dengeli": (95, 1.0, 0.55, 1.1, 1.0, 0.06),
    "Portre": (75, 1.0, 0.95, 1.06, 1.0, 0.04),
    "Belirgin f\u0131r\u00e7a": (150, 1.0, 0.25, 1.14, 1.02, 0.1),
}


def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        rgb = np.array(ImageOps.exif_transpose(source).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_image(path: Path, image: np.ndarray) -> None:
    ext = path.suffix.lower() or ".jpg"
    options = [cv2.IMWRITE_JPEG_QUALITY, 97] if ext in (".jpg", ".jpeg") else []
    ok, buffer = cv2.imencode(ext, image, options)
    if not ok:
        raise ValueError(f"Gorsel kaydedilemedi: {path}")
    buffer.tofile(str(path))


def create_oil_painting(image, sigma_s=75, sigma_r=1.0, saturation=1.06,
                       contrast=1.0, canvas_strength=0.04, detail=0.95,
                       engine="brush", preserve_colors=True, reference=None):
    if engine not in ("brush", "classical", "reference"):
        raise ValueError("Bilinmeyen donusturme yontemi")
    height, width = image.shape[:2]
    factor = min(1.0, 1200 / max(height, width))
    work = cv2.resize(image, (max(1, round(width * factor)), max(1, round(height * factor))), interpolation=cv2.INTER_AREA) if factor < 1 else image
    strength = float(np.clip(sigma_r, 0, 1))
    if strength == 0:
        painted = work.astype(np.float32) / 255
    elif engine == "classical":
        painted = render_classical(work, sigma_s, float(np.clip(detail, 0, 1)), preserve_colors)
    elif engine == "reference":
        style_size = int(np.clip(512 * 85 / max(10, sigma_s), 256, 768))
        painted = render_reference(work, reference, 1.0, detail, preserve_colors, style_size).astype(np.float32) / 255
    else:
        painted = render_strokes(work, float(np.clip(sigma_s, 10, 200)), float(np.clip(detail, 0, 1)))
    if factor < 1:
        painted = cv2.resize(painted, (width, height), interpolation=cv2.INTER_CUBIC)
    original = image.astype(np.float32) / 255.0
    if strength == 0:
        painted = original
    painted = original * (1 - strength) + painted * strength
    rgb = cv2.cvtColor(np.clip(painted * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    color = ImageEnhance.Color(Image.fromarray(rgb)).enhance(saturation)
    color = ImageEnhance.Contrast(color).enhance(contrast)
    return add_canvas_texture(cv2.cvtColor(np.array(color), cv2.COLOR_RGB2BGR), canvas_strength)


def add_canvas_texture(image, strength):
    strength = float(np.clip(strength, 0, 0.45))
    if strength == 0:
        return image
    height, width = image.shape[:2]
    grain = np.random.default_rng(42).normal(0, 1, (height, width)).astype(np.float32)
    strokes = cv2.GaussianBlur(grain, (0, 0), sigmaX=0.6, sigmaY=3.5)
    strokes /= max(float(strokes.std()), 0.001)
    cross_grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=2.5, sigmaY=0.7)
    cross_grain /= max(float(cross_grain.std()), 0.001)
    texture = strokes * 0.08 + cross_grain * 0.025
    texture -= texture.mean()
    relief = 1 + strength * texture
    return np.clip(image.astype(np.float32) * relief[..., None], 0, 255).astype(np.uint8)


class Preview(tk.Canvas):
    def __init__(self, parent, empty_text):
        super().__init__(parent, background="#e9edf0", highlightthickness=0, width=320, height=300)
        self.source = self.photo = self.pending = None
        self.empty_text = empty_text
        self.bind("<Configure>", self.schedule_draw)
        self.bind("<Destroy>", lambda event: self.cancel_draw() if event.widget == self else None)

    def set_image(self, image):
        self.source = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) if image is not None else None
        self.draw()

    def schedule_draw(self, event=None):
        self.cancel_draw()
        self.pending = self.after(80, self.draw)

    def cancel_draw(self):
        if self.pending:
            try:
                self.after_cancel(self.pending)
            except tk.TclError:
                pass
            self.pending = None

    def draw(self):
        self.cancel_draw()
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        if self.source is None:
            self.create_text(width / 2, height / 2, text=self.empty_text, fill="#68747d", font=("Segoe UI", 11), width=max(100, width - 30))
            return
        preview = self.source.copy()
        preview.thumbnail((max(1, width - 24), max(1, height - 24)), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(preview)
        self.create_image(width / 2, height / 2, image=self.photo)


class OilPaintingApp:
    def __init__(self, root):
        self.root = root
        root.title(APP_TITLE)
        root.geometry("1200x780")
        root.minsize(880, 660)
        root.configure(bg="#f6f8fa")
        self.original = self.result = self.input_path = None
        self.busy = False
        self.messages = queue.Queue()
        self.status = tk.StringVar(value="Haz\u0131r")
        self.metadata = tk.StringVar(value="")
        self.preset = tk.StringVar(value=REFERENCE_PRESET)
        self.reference = None
        self.reference_name = tk.StringVar(value="Referans se\u00e7ilmedi")
        self.preserve_colors = tk.BooleanVar(value=True)
        self.values = [tk.DoubleVar(value=value) for value in PRESETS[REFERENCE_PRESET]]
        self.configure_styles()
        self.build_ui()
        reference_path = Path(__file__).with_name("referans_yagli_boya_portre.png")
        if reference_path.is_file():
            try:
                self.reference = read_image(reference_path)
                self.reference_name.set(reference_path.name)
            except Exception:
                pass
        self.update_mode()
        self.poll_id = root.after(100, self.poll_worker)
        root.bind("<Destroy>", self.on_destroy, add="+")

    def configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f6f8fa")
        style.configure("TLabel", background="#f6f8fa", foreground="#25323a", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 21))
        style.configure("Muted.TLabel", foreground="#6b7780", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10), padding=(14, 9), background="#ffffff", borderwidth=1)
        style.configure("Accent.TButton", background="#167c68", foreground="white", font=("Segoe UI Semibold", 11))
        style.map("Accent.TButton", background=[("disabled", "#baccc7"), ("active", "#116653")], foreground=[("disabled", "#f6f8fa")])
        style.configure("Horizontal.TScale", background="#f6f8fa", troughcolor="#dce3e8")
        style.configure("TCombobox", padding=6)
        style.configure("TCheckbutton", background="#f6f8fa", font=("Segoe UI", 9))

    def build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        header = ttk.Frame(self.root, padding=(24, 20, 24, 16))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.open_button = ttk.Button(header, text="Foto\u011fraf a\u00e7", command=self.open_image)
        self.open_button.grid(row=0, column=1, padx=(12, 8))
        self.save_button = ttk.Button(header, text="D\u0131\u015fa aktar", command=self.save_image, state="disabled")
        self.save_button.grid(row=0, column=2)
        ttk.Label(header, textvariable=self.metadata, style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
        body = ttk.Frame(self.root, padding=(24, 0, 24, 12))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        views = ttk.Frame(body)
        views.grid(row=0, column=0, sticky="nsew", padx=(0, 24))
        views.columnconfigure((0, 1), weight=1, uniform="preview")
        views.rowconfigure(1, weight=1)
        ttk.Label(views, text="OR\u0130J\u0130NAL", style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(6, 10))
        ttk.Label(views, text="YA\u011eLI BOYA", style="Muted.TLabel").grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(6, 10))
        self.original_preview = Preview(views, "Foto\u011fraf se\u00e7ilmedi")
        self.original_preview.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.result_preview = Preview(views, "Hen\u00fcz sonu\u00e7 yok")
        self.result_preview.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        sidebar = ttk.Frame(body, width=230)
        sidebar.grid(row=0, column=1, sticky="ns")
        ttk.Label(sidebar, text="Stil", font=("Segoe UI Semibold", 12)).pack(anchor="w", pady=(4, 10))
        self.preset_menu = ttk.Combobox(sidebar, textvariable=self.preset, values=list(PRESETS), state="readonly", width=23)
        self.preset_menu.pack(fill="x", pady=(0, 12))
        self.preset_menu.bind("<<ComboboxSelected>>", self.apply_preset)
        self.local_panel = ttk.Frame(sidebar)
        self.local_panel.pack(fill="x")
        self.reference_panel = ttk.Frame(self.local_panel)
        self.reference_button = ttk.Button(self.reference_panel, text="Referans tablo se\u00e7", command=self.open_reference)
        self.reference_button.pack(fill="x", pady=(0, 6))
        ttk.Label(self.reference_panel, textvariable=self.reference_name, style="Muted.TLabel", wraplength=210).pack(fill="x", pady=(0, 10))
        self.colors_checkbox = ttk.Checkbutton(self.local_panel, text="Foto\u011fraf renklerini koru", variable=self.preserve_colors, command=self.settings_changed)
        self.colors_checkbox.pack(fill="x", pady=(0, 14))
        self.scales = []
        self.setting_rows = []
        settings = [("F\u0131r\u00e7a boyutu", 10, 200), ("Boya yo\u011funlu\u011fu", 0, 1),
                    ("Detay koruma", 0, 1), ("Renk doygunlu\u011fu", 0.5, 1.8),
                    ("Kontrast", 0.7, 1.5), ("Tuval dokusu", 0, 0.45)]
        for index, (label, low, high) in enumerate(settings):
            row = ttk.Frame(self.local_panel)
            row.pack(fill="x")
            ttk.Label(row, text=label).pack(side="left")
            value_label = ttk.Label(row, style="Muted.TLabel", width=5, anchor="e")
            value_label.pack(side="right")
            variable = self.values[index]
            def update_label(*_, var=variable, target=value_label, integer=index == 0):
                target.configure(text=f"{var.get():.0f}" if integer else f"{var.get():.2f}")
            variable.trace_add("write", update_label)
            update_label()
            slider = ttk.Scale(self.local_panel, from_=low, to=high, variable=variable, command=self.settings_changed)
            slider.pack(fill="x", pady=(5, 12))
            self.scales.append(slider)
            self.setting_rows.append(row)
        self.process_button = ttk.Button(sidebar, text="Ya\u011fl\u0131 boyaya d\u00f6n\u00fc\u015ft\u00fcr", style="Accent.TButton", command=self.process_image, state="disabled")
        self.process_button.pack(fill="x", pady=(8, 0))
        footer = ttk.Frame(self.root, padding=(24, 8, 24, 16))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=160)
        self.progress.grid(row=0, column=1, sticky="e")

    def settings_changed(self, _=None):
        if self.result is not None and not self.busy:
            self.status.set("Ayarlar de\u011fi\u015fti. Sonu\u00e7 yeniden d\u00f6n\u00fc\u015ft\u00fcrildi\u011finde g\u00fcncellenir.")

    def apply_preset(self, _=None):
        if self.preset.get() in PRESETS:
            for variable, value in zip(self.values, PRESETS[self.preset.get()]):
                variable.set(value)
        self.update_mode()
        self.settings_changed()

    def update_mode(self):
        reference_mode = self.preset.get() == REFERENCE_PRESET
        self.reference_panel.pack_forget()
        self.colors_checkbox.pack_forget()
        for row, scale in zip(self.setting_rows, self.scales):
            row.pack_forget()
            scale.pack_forget()
        if reference_mode:
            self.reference_panel.pack(fill="x")
        self.colors_checkbox.pack(fill="x", pady=(0, 14))
        for index, (row, scale) in enumerate(zip(self.setting_rows, self.scales)):
            if not reference_mode or index < 3:
                row.pack(fill="x")
                scale.pack(fill="x", pady=(5, 12))
        self.colors_checkbox.configure(state="normal" if reference_mode or self.preset.get() == "Klasik tablo" else "disabled")

    def open_reference(self):
        filename = filedialog.askopenfilename(title="Ya\u011fl\u0131 boya referans\u0131", filetypes=[("G\u00f6rseller", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff")])
        if filename:
            try:
                reference = read_image(Path(filename))
                self.reference = reference
                self.reference_name.set(Path(filename).name)
                self.settings_changed()
            except Exception as exc:
                messagebox.showerror(APP_TITLE, str(exc))

    def open_image(self):
        filename = filedialog.askopenfilename(title="Foto\u011fraf a\u00e7", filetypes=[("G\u00f6rseller", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"), ("T\u00fcm dosyalar", "*.*")])
        if not filename:
            return
        try:
            image = read_image(Path(filename))
            self.input_path = Path(filename)
            self.original = image
            self.result = None
            self.original_preview.set_image(image)
            self.result_preview.set_image(None)
            self.save_button.configure(state="disabled")
            self.process_button.configure(state="normal")
            height, width = image.shape[:2]
            self.metadata.set(f"{self.input_path.name}  |  {width} x {height} px")
            self.status.set("Foto\u011fraf y\u00fcklendi")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def process_image(self):
        if self.original is None or self.busy:
            return
        if self.preset.get() == REFERENCE_PRESET and self.reference is None:
            messagebox.showinfo(APP_TITLE, "\u00d6nce bir ya\u011fl\u0131 boya referans\u0131 se\u00e7in.")
            return
        self.busy = True
        self.status.set("Tablo olu\u015fturuluyor...")
        self.progress.start(12)
        for button in (self.open_button, self.process_button, self.save_button):
            button.configure(state="disabled")
        self.preset_menu.configure(state="disabled")
        self.colors_checkbox.configure(state="disabled")
        self.reference_button.configure(state="disabled")
        for scale in self.scales:
            scale.configure(state="disabled")
        brush, strength, detail, saturation, contrast, texture = [var.get() for var in self.values]
        source = self.original
        engine = "reference" if self.preset.get() == REFERENCE_PRESET else "classical" if self.preset.get() == "Klasik tablo" else "brush"
        preserve_colors = self.preserve_colors.get()
        reference = self.reference
        def worker():
            try:
                result = create_oil_painting(source, brush, strength, saturation, contrast, texture, detail, engine, preserve_colors, reference)
                self.messages.put((result, None))
            except Exception as exc:
                self.messages.put((None, str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def poll_worker(self):
        try:
            result, error = self.messages.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            self.progress.stop()
            self.progress.configure(value=0)
            self.open_button.configure(state="normal")
            self.process_button.configure(state="normal")
            self.preset_menu.configure(state="readonly")
            self.colors_checkbox.configure(state="normal" if self.preset.get() in ("Klasik tablo", REFERENCE_PRESET) else "disabled")
            self.reference_button.configure(state="normal")
            for scale in self.scales:
                scale.configure(state="normal")
            if error:
                self.status.set("D\u00f6n\u00fc\u015ft\u00fcrme tamamlanamad\u0131")
                messagebox.showerror(APP_TITLE, error)
            else:
                self.result = result
                self.result_preview.set_image(result)
                height, width = result.shape[:2]
                self.status.set(f"Tamamland\u0131. {width} x {height} px")
            self.save_button.configure(state="normal" if self.result is not None else "disabled")
        self.poll_id = self.root.after(100, self.poll_worker)

    def on_destroy(self, event):
        if event.widget == self.root:
            try:
                self.root.after_cancel(self.poll_id)
            except tk.TclError:
                pass

    def save_image(self):
        if self.result is None:
            return
        filename = filedialog.asksaveasfilename(title="D\u0131\u015fa aktar", defaultextension=".png", initialfile=f"{self.input_path.stem}_yagli_boya.png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("WebP", "*.webp")])
        if filename:
            try:
                write_image(Path(filename), self.result)
                self.status.set(f"Kaydedildi: {Path(filename).name}")
            except Exception as exc:
                messagebox.showerror(APP_TITLE, str(exc))


def parse_args():
    parser = argparse.ArgumentParser(description="Fotografi yagli boya stiline donusturur.")
    parser.add_argument("input", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--style", choices=["classical", "brush", "reference"], default="reference")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--preserve-colors", action="store_true")
    parser.add_argument("--style-colors", action="store_true", help="Klasik tablonun renk paletini kullan")
    parser.add_argument("--brush", type=float, default=70)
    parser.add_argument("--smoothness", type=float, default=0.96, help="Boya yogunlugu: 0-1")
    parser.add_argument("--detail", type=float, default=0.8, help="Detay koruma: 0-1")
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--texture", type=float, default=0.025)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input:
        source = Path(args.input)
        destination = Path(args.output) if args.output else source.with_name(f"{source.stem}_yagli_boya.png")
        reference = read_image(args.reference) if args.style == "reference" and args.reference else None
        colors = args.preserve_colors if args.style == "reference" else not args.style_colors
        result = create_oil_painting(read_image(source), args.brush, args.smoothness, args.saturation, args.contrast, args.texture, args.detail, args.style, colors, reference)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_image(destination, result)
        print(f"Kaydedildi: {destination}")
    else:
        root = tk.Tk()
        OilPaintingApp(root)
        root.mainloop()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Hata: {error}", file=sys.stderr)
        raise SystemExit(1)
