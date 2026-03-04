from __future__ import annotations

from textual.widgets import Static

from agent.phases import TEACHER_PHASE_ORDER, INTERVIEWER_PHASE_ORDER


class Sidebar(Static):
    """Displays phase list and progress summary."""

    def __init__(self, mode: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._mode = mode
        self._phase_order = (
            TEACHER_PHASE_ORDER if mode == "teacher" else INTERVIEWER_PHASE_ORDER
        )
        self._current_phase: str = self._phase_order[0]
        self._completed_phases: set[str] = set()
        self._progress_data: dict = {}

    def on_mount(self) -> None:
        self._render_content()

    def refresh_state(
        self,
        current_phase: str,
        completed_phases: set[str],
        progress_data: dict | None = None,
    ) -> None:
        self._current_phase = current_phase
        self._completed_phases = completed_phases
        if progress_data is not None:
            self._progress_data = progress_data
        self._render_content()

    def _render_content(self) -> None:
        lines: list[str] = []
        lines.append("[b]PHASES[/b]")
        lines.append("─────────")
        for phase in self._phase_order:
            if phase in self._completed_phases:
                lines.append(f"[dim]  ✓ {phase}[/dim]")
            elif phase == self._current_phase:
                lines.append(f"[bold]  ▸ {phase}[/bold]")
            else:
                lines.append(f"    {phase}")

        lines.append("")
        lines.append("[b]PROGRESS[/b]")
        lines.append("─────────")
        level = self._progress_data.get("level", "—")
        score = self._progress_data.get("score", "—")
        due = self._progress_data.get("due", "—")
        lines.append(f"  Level: {level}")
        lines.append(f"  Score: {score}")
        lines.append(f"  Due:   {due}")

        lines.append("")
        lines.append("[dim]ctrl+p Dashboard[/dim]")
        lines.append("[dim]/help Commands[/dim]")

        self.update("\n".join(lines))


class ToolIndicator(Static):
    """Brief inline indicator during tool execution."""

    DEFAULT_CSS = """
    ToolIndicator {
        color: $text-muted;
        text-style: italic;
        margin: 0 2;
    }
    """

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"⟳ {tool_name}...")
        self._tool_name = tool_name
