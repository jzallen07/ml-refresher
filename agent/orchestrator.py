from __future__ import annotations

from agent.loop import AgentLoop, AgentResponse
from agent.phases import (
    PhaseConfig,
    TEACHER_PHASES,
    INTERVIEWER_PHASES,
    TEACHER_PHASE_ORDER,
    INTERVIEWER_PHASE_ORDER,
)


class SessionOrchestrator:
    def __init__(self, mode: str, topic: str, agent: AgentLoop):
        if mode not in ("teacher", "interviewer"):
            raise ValueError(f"Unknown mode: {mode}")

        self._mode = mode
        self._topic = topic
        self._agent = agent

        if mode == "teacher":
            self._phases = TEACHER_PHASES
            self._phase_order = TEACHER_PHASE_ORDER
        else:
            self._phases = INTERVIEWER_PHASES
            self._phase_order = INTERVIEWER_PHASE_ORDER

        self._current_phase_idx = 0
        self._turn_in_phase = 0
        self._session_complete = False
        self._recent_scores: list[float] = []

        self._context: dict = {
            "questions_asked": [],
            "scores": [],
            "quiz_evaluated": False,
            "questions_remaining": 3,
        }

    @property
    def current_phase(self) -> str:
        return self._phase_order[self._current_phase_idx]

    @property
    def current_config(self) -> PhaseConfig:
        return self._phases[self.current_phase]

    @property
    def is_complete(self) -> bool:
        return self._session_complete

    @property
    def recommended_difficulty(self) -> str:
        if not self._recent_scores:
            return "intermediate"
        avg = sum(self._recent_scores[-5:]) / len(self._recent_scores[-5:])
        if avg >= 0.85:
            return "advanced"
        elif avg >= 0.6:
            return "intermediate"
        return "basic"

    async def start(self) -> AgentResponse:
        self._enter_phase(self.current_phase)
        return await self._run_phase_step()

    async def handle_user_message(self, text: str) -> AgentResponse:
        self._agent.add_user_message(text)
        return await self._run_phase_step()

    def _enter_phase(self, phase: str):
        config = self._phases[phase]
        prompt = config.system_prompt_template.format(topic=self._topic)
        if self._recent_scores:
            avg = sum(self._recent_scores[-5:]) / len(self._recent_scores[-5:])
            prompt += (
                f"\n\n[Session Context]\n"
                f"Recommended difficulty: {self.recommended_difficulty}\n"
                f"Recent average score: {avg:.0%}\n"
                f"Use this to calibrate question difficulty."
            )
        self._agent.set_system_prompt(prompt)
        self._agent.set_available_tools(config.available_tools)
        self._turn_in_phase = 0

    async def _run_phase_step(self) -> AgentResponse:
        config = self.current_config

        # Force a specific tool on the first turn of this phase
        tool_choice = None
        if config.forced_tool and self._turn_in_phase == 0:
            tool_choice = {"type": "tool", "name": config.forced_tool}

        self._turn_in_phase += 1
        response = await self._agent.step(tool_choice=tool_choice)

        # If the model wants to call more tools, keep stepping
        while response.stop_reason == "tool_use":
            response = await self._agent.step()

        self._update_context(response)

        if self._should_transition(config, response):
            self._advance_phase()

        return response

    def _should_transition(self, config: PhaseConfig, response: AgentResponse) -> bool:
        if config.transition_condition == "auto":
            return response.stop_reason == "end_turn"
        elif config.transition_condition == "user_ready":
            return self._turn_in_phase >= config.max_turns
        elif config.transition_condition == "quiz_complete":
            return self._context.get("quiz_evaluated", False)
        return False

    def _advance_phase(self):
        # Interviewer loops: after evaluate, go back to question if more remaining
        if (
            self._mode == "interviewer"
            and self.current_phase == "evaluate"
            and self._context["questions_remaining"] > 0
        ):
            self._context["questions_remaining"] -= 1
            idx = self._phase_order.index("question")
            self._current_phase_idx = idx
            self._enter_phase("question")
            return

        next_idx = self._current_phase_idx + 1
        if next_idx >= len(self._phase_order):
            self._session_complete = True
            return

        self._current_phase_idx = next_idx
        self._enter_phase(self.current_phase)

    def _update_context(self, response: AgentResponse):
        for tool_name in response.tools_called:
            if tool_name == "get_interview_question":
                self._context["questions_asked"].append(tool_name)
            elif tool_name == "evaluate_answer":
                self._context["quiz_evaluated"] = True
                result = response.tool_results_data.get("evaluate_answer", {})
                raw_score = result.get("score")
                if raw_score is not None:
                    normalized = float(raw_score) / 100.0 if float(raw_score) > 1 else float(raw_score)
                    self._recent_scores.append(normalized)
            elif tool_name == "update_progress":
                pass  # Tracked by state layer
