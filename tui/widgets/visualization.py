from __future__ import annotations

import platform
import subprocess

from textual.widgets import Static

try:
    from PIL import Image
    from rich_pixels import Pixels
    from rich.panel import Panel

    _HAS_RICH_PIXELS = True
except ImportError:
    _HAS_RICH_PIXELS = False


class VisualizationWidget(Static):
    """Displays a matplotlib PNG inline and opens it in the system viewer."""

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
        self._open_externally()

    def _open_externally(self) -> None:
        """Open the image in the system's default viewer for full-res viewing."""
        try:
            if platform.system() == "Darwin":
                subprocess.Popen(["open", self._image_path])
            elif platform.system() == "Linux":
                subprocess.Popen(["xdg-open", self._image_path])
        except OSError:
            pass

    def _render_image(self) -> None:
        if not _HAS_RICH_PIXELS:
            self.update(
                f"[dim][Visualization: {self._title} — opened externally][/dim]"
            )
            return

        try:
            img = Image.open(self._image_path)

            # Use full terminal width for maximum detail.
            # Each column = 2 pixels wide (half-block chars), each row = 2 pixels tall.
            term_width = min(self.app.size.width - 4, 160) if self.app else 100
            target_px = term_width * 2
            if img.width > target_px:
                ratio = target_px / img.width
                img = img.resize(
                    (target_px, int(img.height * ratio)),
                    Image.LANCZOS,
                )

            pixels = Pixels.from_image(img)
            panel = Panel(
                pixels,
                title=self._title,
                subtitle="[dim]opened in viewer[/dim]",
                border_style="cyan",
                expand=False,
            )
            self.update(panel)
        except Exception as exc:
            self.update(f"[dim][Could not render visualization: {exc}][/dim]")
