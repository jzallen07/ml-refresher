from __future__ import annotations

from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Center, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static

from cli.api import MLRefresherAPI


@dataclass
class SessionConfig:
    mode: str
    topic: str


class WelcomeScreen(Screen[SessionConfig]):
    CSS = """
    WelcomeScreen {
        align: center middle;
    }

    #welcome-card {
        width: 60;
        height: auto;
        border: solid $primary;
        padding: 2 4;
    }

    #welcome-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #mode-label, #topic-label {
        margin-top: 1;
        margin-bottom: 0;
    }

    #topic-input {
        margin-top: 1;
    }

    #topic-hint {
        color: $text-muted;
        text-style: italic;
        margin-top: 0;
    }

    #start-btn {
        margin-top: 2;
        width: 100%;
    }

    #error-label {
        color: $error;
        margin-top: 1;
        display: none;
    }

    #error-label.visible {
        display: block;
    }
    """

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="welcome-card"):
                yield Static("ML Refresher", id="welcome-title")
                yield Label("Mode", id="mode-label")
                with RadioSet(id="mode-select"):
                    yield RadioButton("Teacher", value=True)
                    yield RadioButton("Interviewer")
                yield Label("Topic", id="topic-label")
                yield Input(placeholder="e.g. attention_mechanisms", id="topic-input")
                yield Static("", id="topic-hint")
                yield Label("", id="error-label")
                yield Button("Start Session", variant="primary", id="start-btn")

    def on_mount(self) -> None:
        try:
            api = MLRefresherAPI()
            topics = api.list_all_topics()
            names = [t.get("id", t.get("topic", "")) for t in topics[:8]]
            if names:
                hint = f"Available: {', '.join(names)}"
                self.query_one("#topic-hint", Static).update(hint)
        except Exception:
            pass

    @on(Button.Pressed, "#start-btn")
    def _submit(self) -> None:
        topic = self.query_one("#topic-input", Input).value.strip()
        if not topic:
            error_label = self.query_one("#error-label", Label)
            error_label.update("Please enter a topic.")
            error_label.add_class("visible")
            return

        radio_set = self.query_one("#mode-select", RadioSet)
        mode = "teacher" if radio_set.pressed_index == 0 else "interviewer"
        self.dismiss(SessionConfig(mode=mode, topic=topic))


class ProgressDashboard(Screen[None]):
    CSS = """
    ProgressDashboard {
        align: center middle;
    }

    #dashboard-scroll {
        width: 80;
        max-height: 90%;
        border: solid $primary;
        padding: 1 2;
    }

    #dashboard-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #dashboard-content {
        margin: 1 0;
    }

    #close-btn {
        margin-top: 1;
        width: 100%;
    }
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("ctrl+p", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Center():
            with VerticalScroll(id="dashboard-scroll"):
                yield Static("Progress Dashboard", id="dashboard-title")
                yield Static("Loading...", id="dashboard-content")
                yield Button("Close", variant="default", id="close-btn")

    def on_mount(self) -> None:
        self._load_data()

    @on(Button.Pressed, "#close-btn")
    def _close(self) -> None:
        self.dismiss(None)

    def _load_data(self) -> None:
        try:
            api = MLRefresherAPI()
            overall = api.get_progress()
            schedule = api.get_review_schedule()

            from cli.state.db import StateDB
            db = StateDB()
            try:
                all_topics = db.get_all_topics()
            finally:
                db.close()

            started_topics = [t for t in all_topics if t["total_interactions"] > 0]
            topic_details = []
            for t in started_topics:
                p = api.get_progress(t["id"])
                topic_details.append(p)

            self._render_dashboard(overall, schedule, topic_details)
        except Exception as e:
            self.query_one("#dashboard-content", Static).update(
                f"[red]Error loading data: {e}[/red]"
            )

    def _render_dashboard(self, overall: dict, schedule: dict, topic_details: list[dict]) -> None:
        lines: list[str] = []

        # Overall Summary
        lines.append("[b]OVERALL SUMMARY[/b]")
        lines.append("─" * 40)
        total = overall.get("total_topics", 0)
        started = overall.get("topics_started", 0)
        lines.append(f"  Topics: {started}/{total} started")
        levels = overall.get("level_counts", {})
        lines.append(
            f"  Levels: {levels.get('novice', 0)} novice, "
            f"{levels.get('intermediate', 0)} intermediate, "
            f"{levels.get('advanced', 0)} advanced"
        )
        avg = overall.get("overall_average")
        lines.append(f"  Average: {avg:.0%}" if avg is not None else "  Average: —")
        weakest = overall.get("weakest_topic")
        strongest = overall.get("strongest_topic")
        if weakest:
            lines.append(f"  Weakest: {weakest}")
        if strongest:
            lines.append(f"  Strongest: {strongest}")

        # Review Schedule
        lines.append("")
        lines.append("[b]REVIEW SCHEDULE[/b]")
        lines.append("─" * 40)
        due_total = schedule.get("total_due", 0)
        lines.append(f"  Cards due: {due_total}")
        cards = schedule.get("cards", [])[:10]
        if cards:
            for c in cards:
                due = c.get("due_date", "?")[:10]
                lines.append(f"  [dim]{c['topic_id']}[/dim] {c['concept']} (due {due})")

        # Topic Details
        if topic_details:
            lines.append("")
            lines.append("[b]TOPIC DETAILS[/b]")
            lines.append("─" * 40)
            lines.append(
                f"  {'Topic':<25} {'Level':<14} {'Score':<8} {'Due':<5} Weak"
            )
            lines.append("  " + "─" * 60)
            for t in topic_details:
                name = (t.get("display_name") or t.get("topic", "?"))[:25]
                level = t.get("level", "—")
                avg = t.get("average_score")
                score_str = f"{avg:.0%}" if avg is not None else "—"
                due = str(t.get("due_cards", 0))
                weak = ", ".join(t.get("weak_concepts", [])[:3]) or "—"
                lines.append(
                    f"  {name:<25} {level:<14} {score_str:<8} {due:<5} [dim]{weak}[/dim]"
                )

        self.query_one("#dashboard-content", Static).update("\n".join(lines))
