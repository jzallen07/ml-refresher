from __future__ import annotations

from textual.widgets import Static

try:
    from PIL import Image
    from rich_pixels import Pixels
    from rich.panel import Panel

    _HAS_RICH_PIXELS = True
except ImportError:
    _HAS_RICH_PIXELS = False


class VisualizationWidget(Static):
    """Displays a matplotlib PNG inline using half-block Unicode characters."""

    DEFAULT_CSS = """
    VisualizationWidget {
        margin: 1 2;
        height: auto;
        background: $surface;
    }
    """

    def __init__(self, image_path: str, title: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._image_path = image_path
        self._title = title

    def on_mount(self) -> None:
        self._render_image()

    def _render_image(self) -> None:
        if not _HAS_RICH_PIXELS:
            self.update(f"[dim][Visualization: {self._title}][/dim]")
            return

        try:
            img = Image.open(self._image_path)

            # Resize to ~80% of terminal width, max 120 columns.
            # Each pixel maps to half a character cell, so target width in
            # pixels is roughly 2x the desired column count.
            term_width = min(self.app.size.width - 6, 120) if self.app else 80
            target_px = term_width * 2
            if img.width > target_px:
                ratio = target_px / img.width
                img = img.resize(
                    (target_px, int(img.height * ratio)),
                    Image.LANCZOS,
                )

            pixels = Pixels.from_image(img)
            panel = Panel(pixels, title=self._title, border_style="cyan", expand=False)
            self.update(panel)
        except Exception as exc:
            self.update(f"[dim][Could not render visualization: {exc}][/dim]")
