from __future__ import annotations

from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Center, Vertical
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
