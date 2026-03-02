from __future__ import annotations

import os

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Header, Footer, Input, Markdown, Static

from tui.widgets import Sidebar, ToolIndicator
from tui.bridge import ToolStarted, ToolFinished, make_callbacks, create_session


class MLRefresherApp(App):
    CSS_PATH = "app.tcss"
    BINDINGS = [("ctrl+q", "quit", "Quit")]

    def __init__(self, mode: str, topic: str, model: str | None = None) -> None:
        self._mode = mode
        self._topic = topic
        self._model = model
        self._orchestrator = None
        self._active_stream = None
        self._completed_phases: set[str] = set()
        super().__init__()
        label = "Teacher" if mode == "teacher" else "Interviewer"
        self.title = f"ML Refresher — {label}"
        self.sub_title = topic

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            yield Sidebar(self._mode, id="sidebar")
            yield VerticalScroll(id="chat-scroll")
        yield Input(placeholder="Type your response...")
        yield Footer()

    def on_mount(self) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            self._show_error("ANTHROPIC_API_KEY environment variable is not set.")
            self.query_one(Input).disabled = True
            return

        on_text, on_tool_start, on_tool_end = make_callbacks(self)
        self._orchestrator = create_session(
            mode=self._mode,
            topic=self._topic,
            model=self._model,
            on_text=on_text,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
        )
        self.start_session()

    @work(exclusive=True)
    async def start_session(self) -> None:
        chat = self.query_one("#chat-scroll", VerticalScroll)
        md_widget = Markdown(classes="agent-message")
        await chat.mount(md_widget)
        chat.anchor()

        stream = Markdown.get_stream(md_widget)
        self._active_stream = stream
        try:
            await self._orchestrator.start()
        except Exception as e:
            self._show_error(f"Error: {e}")
        finally:
            await stream.stop()
            self._active_stream = None
            self._update_sidebar()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        input_widget = self.query_one(Input)
        input_widget.value = ""
        input_widget.disabled = True

        chat = self.query_one("#chat-scroll", VerticalScroll)
        user_msg = Static(text, classes="user-message")
        chat.mount(user_msg)
        chat.anchor()

        self.run_agent(text)

    @work(exclusive=True)
    async def run_agent(self, text: str) -> None:
        chat = self.query_one("#chat-scroll", VerticalScroll)
        md_widget = Markdown(classes="agent-message")
        await chat.mount(md_widget)
        chat.anchor()

        stream = Markdown.get_stream(md_widget)
        self._active_stream = stream
        try:
            await self._orchestrator.handle_user_message(text)
        except Exception as e:
            self._show_error(f"Error: {e}")
        finally:
            await stream.stop()
            self._active_stream = None
            self._update_sidebar()
            input_widget = self.query_one(Input)
            if self._orchestrator and self._orchestrator.is_complete:
                input_widget.placeholder = "Session complete."
                input_widget.disabled = True
            else:
                input_widget.disabled = False
                input_widget.focus()

    def on_tool_started(self, message: ToolStarted) -> None:
        chat = self.query_one("#chat-scroll", VerticalScroll)
        indicator = ToolIndicator(message.name)
        chat.mount(indicator)
        chat.anchor()

    def on_tool_finished(self, message: ToolFinished) -> None:
        indicators = self.query(ToolIndicator)
        for indicator in indicators:
            if indicator._tool_name == message.name:
                indicator.remove()
                break

    def _update_sidebar(self) -> None:
        if not self._orchestrator:
            return
        current = self._orchestrator.current_phase
        order = self._orchestrator._phase_order
        idx = order.index(current)
        self._completed_phases = set(order[:idx])
        self.query_one(Sidebar).refresh_state(
            current_phase=current,
            completed_phases=self._completed_phases,
        )

    def _show_error(self, text: str) -> None:
        chat = self.query_one("#chat-scroll", VerticalScroll)
        chat.mount(Static(text, classes="error-message"))
