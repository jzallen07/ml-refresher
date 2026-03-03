from __future__ import annotations

import asyncio
import os

import anthropic
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Header, Footer, Input, Markdown, Static

from tui.widgets import Sidebar, ToolIndicator
from tui.bridge import ToolStarted, ToolFinished, make_callbacks, create_session


class MLRefresherApp(App):
    CSS_PATH = "app.tcss"
    BINDINGS = [("ctrl+q", "quit", "Quit")]

    def __init__(
        self,
        mode: str | None = None,
        topic: str | None = None,
        model: str | None = None,
    ) -> None:
        self._mode = mode
        self._topic = topic
        self._model = model
        self._orchestrator = None
        self._api = None
        self._active_stream = None
        self._completed_phases: set[str] = set()
        super().__init__()
        if mode and topic:
            label = "Teacher" if mode == "teacher" else "Interviewer"
            self.title = f"ML Refresher — {label}"
            self.sub_title = topic

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            yield Sidebar(self._mode or "teacher", id="sidebar")
            yield VerticalScroll(id="chat-scroll")
        yield Input(placeholder="Type your response...")
        yield Footer()

    def on_mount(self) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            self._show_error("ANTHROPIC_API_KEY environment variable is not set.")
            self.query_one(Input).disabled = True
            return

        if self._mode and self._topic:
            self._initialize_session()
        else:
            from tui.screens import WelcomeScreen
            self.push_screen(WelcomeScreen(), callback=self._on_welcome_complete)

    def _on_welcome_complete(self, config) -> None:
        self._mode = config.mode
        self._topic = config.topic
        label = "Teacher" if self._mode == "teacher" else "Interviewer"
        self.title = f"ML Refresher — {label}"
        self.sub_title = self._topic
        # Replace sidebar with correct mode
        container = self.query_one("#main-container", Horizontal)
        old_sidebar = self.query_one(Sidebar)
        new_sidebar = Sidebar(self._mode, id="sidebar")
        container.mount(new_sidebar, before=old_sidebar)
        old_sidebar.remove()
        self._initialize_session()

    def _initialize_session(self) -> None:
        on_text, on_tool_start, on_tool_end = make_callbacks(self)
        self._orchestrator, self._api = create_session(
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
            await asyncio.wait_for(self._orchestrator.start(), timeout=120.0)
        except asyncio.TimeoutError:
            self._show_error("Session timed out. Please try again.")
            self.query_one(Input).disabled = False
        except anthropic.AuthenticationError:
            self._show_error("Authentication failed. Check your ANTHROPIC_API_KEY.")
            self.query_one(Input).disabled = True
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
            await asyncio.wait_for(
                self._orchestrator.handle_user_message(text), timeout=120.0
            )
        except asyncio.TimeoutError:
            self._show_error("Response timed out. Please try again.")
        except anthropic.AuthenticationError:
            self._show_error("Authentication failed. Check your ANTHROPIC_API_KEY.")
            self.query_one(Input).disabled = True
            return
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

        if "error" in message.result:
            chat = self.query_one("#chat-scroll", VerticalScroll)
            chat.mount(Static(f"⚠ {message.name}: {message.result['error']}", classes="tool-error"))

        self._update_sidebar()

    def _update_sidebar(self) -> None:
        if not self._orchestrator:
            return
        current = self._orchestrator.current_phase
        order = self._orchestrator._phase_order
        idx = order.index(current)
        self._completed_phases = set(order[:idx])

        progress_data = None
        if self._api and self._topic:
            try:
                raw = self._api.get_progress(self._topic)
                progress_data = {
                    "level": raw.get("level", "—"),
                    "score": f"{raw['score']:.0%}" if "score" in raw else "—",
                    "due": str(raw.get("due_count", raw.get("due", "—"))),
                }
            except Exception:
                pass

        self.query_one(Sidebar).refresh_state(
            current_phase=current,
            completed_phases=self._completed_phases,
            progress_data=progress_data,
        )

    def _show_error(self, text: str) -> None:
        chat = self.query_one("#chat-scroll", VerticalScroll)
        chat.mount(Static(text, classes="error-message"))
