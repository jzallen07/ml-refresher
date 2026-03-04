# PRD: Interactive ML Tutor TUI

## 1. Overview

### Problem
The ML Refresher repository contains 10 progressive PyTorch lessons, 50+ LLM interview questions across 22 topics, and detailed explanatory content — but the only way to consume it is passively reading through Jupyter Book or running notebooks manually. There's no guided learning path, no adaptive difficulty, no way to test understanding, and no interview simulation.

### Solution
Build a Terminal User Interface (TUI) powered by an agentic LLM that operates in two personas:

1. **Teacher**: A Socratic tutor that delivers guided, interactive learning sessions grounded in the book's content. Not a chatbot you ask questions to — a teacher that drives the lesson, adapts to your level, and uses tools to demonstrate concepts.

2. **Interviewer**: A mock ML interviewer that draws from the existing 50+ question bank, evaluates answers against structured rubrics, asks realistic follow-up probes, and provides scored feedback.

Both personas share a persistent knowledge state (what the user knows, where they struggle) so learning carries across sessions.

### Non-Goals
- Web UI or browser-based interface (Jupyter Book already serves this)
- Multi-user support or authentication
- Fine-tuning or training custom models
- Supporting LLM providers other than Anthropic (v1; can abstract later)

---

## 2. Architecture

### 2.1 Single Agent, Dual Persona

Use a single LLM agent with persona switching via system prompt — not a multi-agent system.

**Rationale** (from research):
- Cognition (Devin team): multi-agent architectures create "fragile systems due to poor context sharing and conflicting decisions"
- The two modes are mutually exclusive (user is in Teacher OR Interviewer mode at any time), so there's nothing to parallelize
- Single agent means lower latency, simpler state management, and consistent context within a session

**Implementation**: Swap the system prompt when the user switches modes. Carry over the knowledge state but start a fresh conversation history.

### 2.2 Agent Framework: Custom, Not Off-the-Shelf

**Decision: We write our own lightweight agent loop. No LangChain, LlamaIndex, CrewAI, or other framework.**

**Why not an existing framework?**

| Concern | Detail |
|---------|--------|
| **Fixed, small tool set** | We have 11 tools total. Frameworks like LangChain add value when you need a plugin ecosystem of hundreds of tools, dynamic tool discovery, or multi-provider abstraction. We don't. |
| **Orchestration is the product** | The phase-based workflow, forced tool calls, and tool gating per phase ARE the core product logic. Frameworks abstract away exactly the control surface we need to own. Delegating orchestration to a framework means fighting it when we need custom phase transitions. |
| **Anthropic API is sufficient** | The `anthropic` Python SDK provides streaming, tool use, `tool_choice` for forced calls, and structured tool schemas natively. A framework on top adds indirection without capability. |
| **Debuggability** | When a session goes wrong (agent skips the quiz, gives a direct answer instead of Socratic questioning), we need to inspect the exact API call — system prompt, tool list, tool_choice setting, conversation history. Frameworks insert opaque layers between us and the API. |
| **Dependency weight** | LangChain alone pulls in 50+ transitive dependencies. Our entire agent layer is ~500 lines of Python wrapping the `anthropic` SDK. |

This aligns with Anthropic's own engineering guidance: for applications with a known tool set and well-defined workflows, a simple while-loop with tool dispatch outperforms framework-based approaches.

**What we build ourselves:**
- Agentic while-loop with tool dispatch (~100 lines)
- Phase-based orchestration / session state machine (~200 lines)
- Tool registry with per-phase gating (~100 lines)
- Streaming bridge from Anthropic API to Textual `MarkdownStream` (~100 lines)

**What we use libraries for (not frameworks):**
- `anthropic` SDK — raw API access (streaming, tool use, messages)
- `lancedb` — vector storage with native hybrid search (not an agent framework, just a database)
- `sentence-transformers` — embedding computation
- `py-fsrs` — spaced repetition algorithm
- `textual` — TUI rendering

The distinction: libraries do one thing and we call them. Frameworks call us and impose their abstractions.

### 2.3 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| TUI Framework | Textual v4+ | Streaming Markdown rendering (purpose-built in v4), CSS layouts, async-native, Rich integration |
| LLM | Anthropic Claude API (`anthropic` SDK) | Native tool use, `tool_choice` for forced calls, streaming, structured schemas |
| Agent Loop | Custom while-loop with tool dispatch | Full control over orchestration; no framework overhead |
| Orchestration | Custom phase-based state machine | Enforces session structure, tool gating, forced tool calls per phase |
| Vector Store | LanceDB (embedded) | Native hybrid search (BM25 + dense + RRF), SQL filtering, serverless, no process |
| Embeddings | nomic-embed-text-v1.5 (256d Matryoshka) | 8192 context, 53.01 retrieval NDCG, instruction prefixes, 98% quality at 256d |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (ONNX) | 22M params, ~45ms for 20 candidates on CPU, +17 pts precision |
| Spaced Repetition | py-fsrs | State-of-the-art open-source algorithm (used by Anki), Python native |
| State Persistence | SQLite | Single-file, portable, no server, good for structured learning state |
| CLI Framework | Click 8+ | Subcommand pattern, `--json` flag, composable, lightweight |
| CLI API | `MLRefresherAPI` (custom) | Programmatic interface to CLI ops; called by agent tool shims in-process |
| Code Execution | subprocess with timeout | User runs on own machine; educational code is low-risk |
| Package Manager | uv (existing) | Already used by the project |

### 2.4 High-Level Architecture

The system has three distinct layers: the **TUI** (presentation), the **Agent** (orchestration + LLM), and the **CLI** (repository operations). The CLI is a standalone tool that encapsulates all repository access — content retrieval, code execution, assessment, state management. The agent's tools are thin wrappers that invoke CLI commands. Humans can also use the CLI directly.

```
┌─────────────────────────────────────────────────────────────┐
│                        Textual TUI                           │
│  ┌──────────┐  ┌──────────────────────────────────────────┐ │
│  │ Sidebar  │  │              Chat Area                   │ │
│  │ ──────── │  │  [Streaming Markdown messages]           │ │
│  │ Topics   │  │                                          │ │
│  │ Progress │  │  > Agent response streaming...           │ │
│  │ Mode     │  │                                          │ │
│  │          │  │  ──────────────────────────────────────  │ │
│  │          │  │  [Input]                                 │ │
│  └──────────┘  └──────────────────────┬───────────────────┘ │
└───────────────────────────────────────┼─────────────────────┘
                                        │
                                   ┌────▼─────┐
                                   │ Session  │
                                   │ Orchest. │ ◄── Phase state machine
                                   └────┬─────┘
                                        │
                              ┌─────────▼──────────┐
                              │    Agent Loop       │
                              │  ┌───────────────┐  │
                              │  │ Anthropic API │  │
                              │  │ - streaming   │  │
                              │  │ - tool_choice │  │
                              │  │ - tool use    │  │
                              │  └───────┬───────┘  │
                              │          │          │
                              │  ┌───────▼───────┐  │
                              │  │  Tool Shims   │  │  Thin wrappers: deserialize
                              │  │  (per phase)  │  │  tool_input → call CLI → return JSON
                              │  └───────┬───────┘  │
                              └──────────┼──────────┘
                                         │ invokes
                  ╔══════════════════════╪══════════════════════╗
                  ║          mlr CLI     │                      ║
                  ║    ┌─────────────────▼──────────────────┐   ║
                  ║    │  mlr <command> [args] [--json]     │   ║
                  ║    │                                    │   ║
                  ║    │  content:   search, lesson, question│   ║
                  ║    │  code:      run, example            │   ║
                  ║    │  assess:    quiz, evaluate, next-q  │   ║
                  ║    │  progress:  show, update, review    │   ║
                  ║    │  present:   diagram                 │   ║
                  ║    │  index:     build, status            │   ║
                  ║    └───────┬────────────┬───────────────┘   ║
                  ║            │            │                    ║
                  ║    ┌───────▼──┐   ┌────▼─────┐              ║
                  ║    │ LanceDB  │   │  SQLite   │              ║
                  ║    │ + embeds │   │  + FSRS   │              ║
                  ║    └──────────┘   └──────────┘              ║
                  ╚═════════════════════════════════════════════╝
                        ▲
                        │ also usable directly
                        │
                   Human / scripts / CI
```

**Key architectural property**: The CLI is the single point of access to the repository's content, state, and operations. Neither the TUI nor the agent bypass it. This means:
- The CLI is independently testable without any LLM
- Humans can use `mlr` directly for quick lookups, running quizzes, checking progress
- The agent's tool shims are ~5 lines each (parse input, call CLI, return output)
- Swapping LLM providers only affects the agent layer; the CLI is untouched

---

## 3. Orchestration Layer

The orchestration layer is the critical piece that turns a chatbot into an agent with enforced behavior. The application code — not the LLM — controls what the agent can do at each step.

### 3.1 Core Principle: App Drives, LLM Acts

The LLM is creative within a constrained phase. It chooses how to phrase questions, what analogies to use, how to follow up. But the phase transitions, required tool calls, and available tools are enforced by the orchestration layer in application code.

```
┌─────────────────────────────────────────────────────────┐
│                 Session Orchestrator                     │
│                                                         │
│  Owns:                        Delegates to LLM:         │
│  - Phase transitions          - Natural language output  │
│  - Tool availability          - Which optional tools     │
│  - Forced tool calls          - Follow-up questions      │
│  - Phase-specific prompts     - Analogies & explanations │
│  - Validation of results      - Conversation flow        │
│  - Session state              - Adaptive tone            │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Phase State Machine — Teacher

Each phase defines: a scoped system prompt, available tools, forced tool calls (if any), and transition conditions.

```
┌──────────┐    auto     ┌───────────┐   auto    ┌─────────┐
│ WARMUP   │ ──────────► │ INTRODUCE │ ────────► │ EXPLORE │
│          │             │           │           │         │
│ forced:  │             │ forced:   │           │ tools:  │
│ get_     │             │ search_   │           │ search_ │
│ review_  │             │ content,  │           │ content,│
│ schedule │             │ get_      │           │ run_    │
│          │             │ lesson    │           │ python, │
└──────────┘             └───────────┘           │ get_    │
                                                 │ code_ex │
                                                 └────┬────┘
                                                      │ user or agent
                                                      │ signals ready
                                                 ┌────▼────┐    auto    ┌────────┐
                                                 │PRACTICE │ ─────────► │ WRAPUP │
                                                 │         │            │        │
                                                 │ forced: │            │ forced:│
                                                 │ generate│            │ update_│
                                                 │ _quiz   │            │ progr. │
                                                 └─────────┘            └────────┘
```

**Implementation:**

```python
from enum import Enum
from dataclasses import dataclass

class TeacherPhase(Enum):
    WARMUP = "warmup"
    INTRODUCE = "introduce"
    EXPLORE = "explore"
    PRACTICE = "practice"
    WRAPUP = "wrapup"

@dataclass
class PhaseConfig:
    system_prompt: str              # Scoped prompt for this phase only
    available_tools: list[str]      # Tools the LLM can see
    forced_tool: str | None         # Tool the LLM MUST call first (via tool_choice)
    max_turns: int                  # Max LLM round-trips before auto-transition
    transition_condition: str       # "auto" | "user_ready" | "quiz_complete"

TEACHER_PHASES: dict[TeacherPhase, PhaseConfig] = {
    TeacherPhase.WARMUP: PhaseConfig(
        system_prompt=WARMUP_PROMPT,
        available_tools=["get_review_schedule", "get_question", "evaluate_answer"],
        forced_tool="get_review_schedule",
        max_turns=4,
        transition_condition="auto",
    ),
    TeacherPhase.INTRODUCE: PhaseConfig(
        system_prompt=INTRODUCE_PROMPT,
        available_tools=["search_content", "get_lesson", "render_diagram"],
        forced_tool="search_content",
        max_turns=3,
        transition_condition="auto",
    ),
    TeacherPhase.EXPLORE: PhaseConfig(
        system_prompt=EXPLORE_PROMPT,
        available_tools=["search_content", "run_python", "get_code_example", "render_diagram"],
        forced_tool=None,  # LLM chooses freely
        max_turns=15,
        transition_condition="user_ready",
    ),
    TeacherPhase.PRACTICE: PhaseConfig(
        system_prompt=PRACTICE_PROMPT,
        available_tools=["generate_quiz", "evaluate_answer", "run_python", "search_content"],
        forced_tool="generate_quiz",
        max_turns=10,
        transition_condition="quiz_complete",
    ),
    TeacherPhase.WRAPUP: PhaseConfig(
        system_prompt=WRAPUP_PROMPT,
        available_tools=["update_progress"],
        forced_tool="update_progress",
        max_turns=2,
        transition_condition="auto",
    ),
}
```

### 3.3 Phase State Machine — Interviewer

```
┌──────────┐    auto     ┌──────────┐   per question   ┌──────────┐
│  SETUP   │ ──────────► │ QUESTION │ ◄──────────────── │ EVALUATE │
│          │             │          │ ──────────────────►│          │
│ forced:  │             │ forced:  │                    │ forced:  │
│ get_     │             │ get_     │                    │ evaluate │
│ progress │             │ interview│                    │ _answer  │
│          │             │ _question│                    │          │
└──────────┘             └──────────┘                    └─────┬────┘
                                                               │ all questions done
                                                         ┌─────▼────┐
                                                         │ DEBRIEF  │
                                                         │          │
                                                         │ forced:  │
                                                         │ update_  │
                                                         │ progress │
                                                         └──────────┘
```

```python
class InterviewPhase(Enum):
    SETUP = "setup"
    QUESTION = "question"
    FOLLOWUP = "followup"
    EVALUATE = "evaluate"
    DEBRIEF = "debrief"

INTERVIEW_PHASES: dict[InterviewPhase, PhaseConfig] = {
    InterviewPhase.SETUP: PhaseConfig(
        system_prompt=INTERVIEW_SETUP_PROMPT,
        available_tools=["get_progress"],
        forced_tool="get_progress",
        max_turns=2,
        transition_condition="auto",
    ),
    InterviewPhase.QUESTION: PhaseConfig(
        system_prompt=INTERVIEW_QUESTION_PROMPT,
        available_tools=["get_interview_question", "search_content"],
        forced_tool="get_interview_question",
        max_turns=2,
        transition_condition="auto",
    ),
    InterviewPhase.FOLLOWUP: PhaseConfig(
        system_prompt=INTERVIEW_FOLLOWUP_PROMPT,
        available_tools=["search_content"],
        forced_tool=None,
        max_turns=4,
        transition_condition="user_ready",
    ),
    InterviewPhase.EVALUATE: PhaseConfig(
        system_prompt=INTERVIEW_EVALUATE_PROMPT,
        available_tools=["evaluate_answer", "search_content"],
        forced_tool="evaluate_answer",
        max_turns=2,
        transition_condition="auto",
    ),
    InterviewPhase.DEBRIEF: PhaseConfig(
        system_prompt=INTERVIEW_DEBRIEF_PROMPT,
        available_tools=["update_progress", "get_progress"],
        forced_tool="update_progress",
        max_turns=3,
        transition_condition="auto",
    ),
}
```

### 3.4 Session Orchestrator Implementation

The orchestrator is a ~200-line class that runs the phase loop and delegates to the agent loop within each phase.

```python
class SessionOrchestrator:
    """Drives the session. The LLM acts within phases; this class owns transitions."""

    def __init__(self, mode: str, topic: str, agent: AgentLoop):
        self.mode = mode
        self.topic = topic
        self.agent = agent
        self.phases = TEACHER_PHASES if mode == "teacher" else INTERVIEW_PHASES
        self.phase_order = list(self.phases.keys())
        self.current_phase_idx = 0
        self.session_context = {}  # Shared state across phases (scores, etc.)

    async def run_session(self):
        """Run the full session through all phases."""
        while self.current_phase_idx < len(self.phase_order):
            phase = self.phase_order[self.current_phase_idx]
            config = self.phases[phase]

            result = await self.run_phase(phase, config)

            # Handle phase-specific transition logic
            if self.mode == "interview" and phase == InterviewPhase.EVALUATE:
                if self.session_context["questions_remaining"] > 0:
                    # Loop back to QUESTION phase
                    self.current_phase_idx = self.phase_order.index(InterviewPhase.QUESTION)
                    continue

            self.current_phase_idx += 1

    async def run_phase(self, phase, config: PhaseConfig):
        """Run a single phase: set up constraints, run agent loop, validate."""

        # 1. Configure agent for this phase
        self.agent.set_system_prompt(config.system_prompt)
        self.agent.set_available_tools(config.available_tools)

        # 2. Force first tool call if required
        tool_choice = None
        if config.forced_tool:
            tool_choice = {"type": "tool", "name": config.forced_tool}

        # 3. Run agent loop within phase constraints
        turns = 0
        while turns < config.max_turns:
            response = await self.agent.step(
                tool_choice=tool_choice if turns == 0 else None
            )

            # 4. Validate: did the agent use required tools?
            if config.forced_tool and turns == 0:
                if config.forced_tool not in response.tools_called:
                    # Re-run with explicit force (shouldn't happen with tool_choice, but safety net)
                    continue

            # 5. Check transition condition
            if self._should_transition(config, response):
                break

            turns += 1

        return self.session_context

    def _should_transition(self, config: PhaseConfig, response) -> bool:
        match config.transition_condition:
            case "auto":
                return response.stop_reason == "end_turn"
            case "user_ready":
                return self._user_signaled_ready(response)
            case "quiz_complete":
                return self.session_context.get("quiz_evaluated", False)
```

### 3.5 Enforcement Mechanisms Summary

| Mechanism | What It Enforces | How |
|-----------|-----------------|-----|
| **Phase state machine** | Session structure (warmup → intro → explore → practice → wrapup) | Application code controls phase transitions; LLM cannot skip or reorder phases |
| **Tool gating** | Agent can only use phase-appropriate tools | Each phase declares `available_tools`; only those tool schemas are sent in the API call |
| **Forced tool calls** | Critical actions always happen (e.g., must check review schedule at warmup) | Anthropic API `tool_choice: {"type": "tool", "name": "..."}` on first turn of a phase |
| **Max turns per phase** | Prevents runaway phases (e.g., endless Socratic questioning) | `max_turns` counter; auto-transition when exceeded |
| **Result validation** | Tool outputs are well-formed and within bounds | Application inspects tool call results before appending to conversation (e.g., score 0-100) |
| **Scoped system prompts** | LLM only has instructions for the current phase | Instead of one giant prompt with "step 1, step 2...", each phase gets a focused prompt |
| **Socratic guardrail** | Teacher doesn't give direct answers prematurely | Application monitors agent output; if it detects a long explanation without a question mark in Teacher/EXPLORE phase, injects a system reminder |
| **Conversation injection** | Steer the agent back on track without user seeing it | Insert `role: "user"` messages with `[system]` prefix (invisible to the TUI user) to course-correct |

### 3.6 The Agent Loop (Inner Loop)

The agent loop is the inner loop that runs within each phase. It handles the Anthropic API call, streaming, and tool dispatch.

```python
class AgentLoop:
    """Thin wrapper around the Anthropic API. No orchestration logic here."""

    def __init__(self, client: anthropic.AsyncAnthropic, model: str):
        self.client = client
        self.model = model
        self.system_prompt = ""
        self.messages: list[dict] = []
        self.tool_schemas: list[dict] = []
        self.tool_registry: dict[str, Callable] = {}

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def set_available_tools(self, tool_names: list[str]):
        self.tool_schemas = [
            self.tool_registry[name].schema for name in tool_names
        ]

    async def step(self, tool_choice=None) -> AgentResponse:
        """Single round-trip: send messages → get response → execute tools → return."""
        response = await self.client.messages.create(
            model=self.model,
            system=self.system_prompt,
            messages=self.messages,
            tools=self.tool_schemas,
            tool_choice=tool_choice or {"type": "auto"},
            stream=True,
        )

        result = AgentResponse()

        async for event in response:
            if event.type == "content_block_start" and event.content_block.type == "text":
                yield_to_ui(event)  # Stream text to Textual MarkdownStream
            elif event.type == "content_block_start" and event.content_block.type == "tool_use":
                tool_name = event.content_block.name
                tool_input = await collect_tool_input(event)

                # Execute the tool
                tool_result = await self.tool_registry[tool_name].execute(tool_input)

                # Validate result
                tool_result = self._validate_result(tool_name, tool_result)

                # Append tool use + result to messages
                self.messages.append({"role": "assistant", "content": [tool_use_block]})
                self.messages.append({"role": "user", "content": [tool_result_block]})

                result.tools_called.append(tool_name)

        result.stop_reason = response.stop_reason
        return result

    def _validate_result(self, tool_name: str, result: dict) -> dict:
        """Validate tool results before feeding back to the LLM."""
        if tool_name == "evaluate_answer":
            assert 0 <= result["score"] <= 100, f"Score out of range: {result['score']}"
            assert len(result["concepts_covered"]) >= 0
        if tool_name == "generate_quiz":
            assert len(result["questions"]) > 0, "Quiz must have at least one question"
        return result
```

---

## 4. The `mlr` CLI

The CLI is the backbone of the system. It encapsulates **all** repository operations — content retrieval, code execution, assessment, state management, and indexing — behind a single command-line interface. The agent calls the CLI; humans can also call the CLI directly.

### 4.1 Design Principles

1. **Dual output modes**: Every command supports `--json` for machine consumption (the agent) and formatted human-readable output by default (for direct use).
2. **Stateless commands**: Each invocation is a standalone operation. State lives in SQLite/LanceDB, not in the CLI process. This means the agent can call commands in any order.
3. **Single binary, subcommand pattern**: `mlr <group> <command> [args] [flags]`, following the `git`/`gh`/`docker` convention.
4. **Exit codes**: 0 for success, 1 for user error (bad args, not found), 2 for system error (DB unavailable, index missing). The agent can branch on exit codes.
5. **No LLM dependency**: The CLI never calls the Anthropic API. `generate_quiz` and `evaluate_answer` are exceptions — these are agent-side tools that use the LLM, not CLI commands (see Section 5).

### 4.2 Command Reference

#### Content Commands

```bash
# Search all content via RAG (vector similarity)
mlr content search "how does multi-head attention work" [--topic attention_mechanisms] [--limit 5] [--json]

# Get a full lesson
mlr content lesson pytorch/06 [--json]
mlr content lesson interview/02 [--json]

# Get a specific interview question with rubric
mlr content question attention_q2 [--json]

# List all available topics
mlr content topics [--category pytorch|interview] [--json]

# List all questions in a topic
mlr content questions --topic attention_mechanisms [--json]
```

**Examples:**

```bash
$ mlr content search "what is scaled dot-product attention" --topic attention_mechanisms --limit 3
╭─ Search Results (3 of 12 matches) ──────────────────────────────────────╮
│                                                                          │
│  1. interview_questions/02_attention_mechanisms/README.md (score: 0.92)  │
│     ...Scaled dot-product attention computes attention(Q,K,V) =          │
│     softmax(QK^T / sqrt(d_k))V. The scaling factor 1/sqrt(d_k)          │
│     prevents the dot products from growing too large...                  │
│                                                                          │
│  2. interview_questions/22_transformer_deep_dive/README.md (score: 0.87)│
│     ...The attention function maps a query and a set of key-value        │
│     pairs to an output. The output is a weighted sum of the values...    │
│                                                                          │
│  3. pytorch_refresher/... (score: 0.71)                                 │
│     ...                                                                  │
╰──────────────────────────────────────────────────────────────────────────╯

$ mlr content search "scaled dot-product attention" --json
{
  "chunks": [
    {
      "text": "Scaled dot-product attention computes...",
      "source_file": "interview_questions/02_attention_mechanisms/README.md",
      "topic": "attention_mechanisms",
      "lesson_id": "interview/02",
      "relevance_score": 0.92
    },
    ...
  ]
}
```

```bash
$ mlr content lesson pytorch/06
╭─ Lesson: Linear Regression ─────────────────────────────────────────────╮
│                                                                          │
│  Learning Objectives:                                                    │
│  • Build a linear regression model from scratch in PyTorch              │
│  • Understand the training loop: forward pass, loss, backward, update   │
│  • Visualize model predictions vs ground truth                          │
│                                                                          │
│  Content: (423 lines)                                                   │
│  Code Examples: 3 (linear_model, training_loop, visualization)          │
╰──────────────────────────────────────────────────────────────────────────╯

$ mlr content lesson pytorch/06 --json
{
  "title": "Linear Regression",
  "lesson_id": "pytorch/06",
  "learning_objectives": ["Build a linear regression model..."],
  "content_markdown": "# Linear Regression\n\n...",
  "code_examples": [
    {"name": "linear_model", "code": "import torch...", "description": "Basic linear model class"},
    ...
  ]
}
```

#### Code Commands

```bash
# Run Python code (from string or file)
mlr code run --code "import torch; print(torch.tensor([1,2,3]).shape)" [--timeout 30] [--json]
mlr code run --file /path/to/script.py [--timeout 30] [--json]

# Get a code example from a lesson
mlr code example pytorch/06 [--name training_loop] [--json]
```

**Examples:**

```bash
$ mlr code run --code "import torch; t = torch.randn(3, 4); print(t.shape, t.dtype)"
torch.Size([3, 4]) torch.float32
(exit 0, 42ms)

$ mlr code run --code "import torch; t = torch.randn(3, 4); print(t.shape)" --json
{
  "stdout": "torch.Size([3, 4]) torch.float32\n",
  "stderr": "",
  "exit_code": 0,
  "execution_time_ms": 42,
  "figures": []
}

$ mlr code example pytorch/06 --name training_loop
╭─ Code Example: training_loop ───────────────────────────────────────────╮
│  Source: pytorch_refresher/06_linear_regression/lesson.py               │
│                                                                          │
│  # Training loop                                                        │
│  for epoch in range(num_epochs):                                        │
│      y_pred = model(X)                                                  │
│      loss = criterion(y_pred, y)                                        │
│      optimizer.zero_grad()                                              │
│      loss.backward()                                                    │
│      optimizer.step()                                                   │
╰──────────────────────────────────────────────────────────────────────────╯
```

#### Progress Commands

```bash
# View learning progress
mlr progress show [--topic attention_mechanisms] [--json]

# Record a learning event
mlr progress update --topic attention_mechanisms --event quiz --score 0.85 \
    [--concepts "multi-head attention,scaled dot-product"] [--json]

# Check spaced repetition review schedule
mlr progress review [--json]
```

**Examples:**

```bash
$ mlr progress show
╭─ Learning Progress ─────────────────────────────────────────────────────╮
│                                                                          │
│  Overall: 34% (11/32 topics started)                                    │
│                                                                          │
│  PyTorch Refresher:                                                     │
│  ✅ Tensors ●●● (advanced)     ✅ Reshaping ●●○ (intermediate)         │
│  ✅ Indexing ●●○ (intermediate) ⬜ Math Ops ○○○                         │
│  ⬜ Gradients ○○○              ⬜ Lin. Regression ○○○                   │
│  ...                                                                    │
│                                                                          │
│  Interview Questions:                                                   │
│  ✅ Tokenization ●○○ (novice)  ✅ Attention ●●○ (intermediate)         │
│  ⬜ Transformer Arch ○○○       ⬜ Context & Memory ○○○                  │
│  ...                                                                    │
│                                                                          │
│  Weakest: Loss Functions, Gradients, Model Architectures                │
│  Strongest: Tensors, Attention, Tokenization                            │
╰──────────────────────────────────────────────────────────────────────────╯

$ mlr progress review --json
{
  "due_items": [
    {"topic": "attention_mechanisms", "concept": "scaled dot-product", "days_overdue": 3, "last_score": 0.7},
    {"topic": "tokenization", "concept": "BPE algorithm", "days_overdue": 1, "last_score": 0.9}
  ],
  "total_due": 2
}
```

#### Diagram Commands

```bash
# Render an architecture diagram
mlr diagram show transformer_full [--annotate d_model=512,num_heads=8] [--json]

# List available diagrams
mlr diagram list
```

#### Index Commands

```bash
# Build/rebuild the content index
mlr index build [--force]

# Check index status (is it built? how many chunks? last built?)
mlr index status [--json]
```

### 4.3 CLI Implementation

```python
# cli/__main__.py — entry point: `uv run python -m cli` or `mlr` via pyproject.toml script
import click

@click.group()
def mlr():
    """ML Refresher CLI — content search, code execution, progress tracking."""
    pass

# Register subcommand groups
mlr.add_command(content_group, "content")
mlr.add_command(code_group, "code")
mlr.add_command(progress_group, "progress")
mlr.add_command(diagram_group, "diagram")
mlr.add_command(index_group, "index")
```

```python
# cli/content.py
import click
import json
from cli.services.retriever import Retriever
from cli.services.lessons import LessonLoader

@click.group("content")
def content_group():
    """Content retrieval commands."""
    pass

@content_group.command("search")
@click.argument("query")
@click.option("--topic", default=None, help="Filter to a specific topic.")
@click.option("--limit", default=5, help="Max results.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def search(query: str, topic: str | None, limit: int, as_json: bool):
    """Search book content via semantic similarity."""
    retriever = Retriever()
    results = retriever.search(query, topic_filter=topic, max_results=limit)

    if as_json:
        click.echo(json.dumps({"chunks": [r.to_dict() for r in results]}))
    else:
        # Rich-formatted human output
        render_search_results(results)
```

**Programmatic access**: The CLI commands are also importable as Python functions. The agent tool shims call the functions directly (no subprocess for in-process calls) but the interface contract is identical:

```python
# cli/api.py — programmatic interface (same logic as CLI commands, no Click dependency)
class MLRefresherAPI:
    """Programmatic access to all CLI operations. Used by agent tool shims."""

    def __init__(self, db_path: str = None, index_path: str = None):
        self.db = StateDB(db_path or DEFAULT_DB_PATH)
        self.retriever = Retriever(index_path or DEFAULT_INDEX_PATH)
        self.lessons = LessonLoader()

    def search_content(self, query: str, topic_filter: str = None, max_results: int = 5) -> dict:
        results = self.retriever.search(query, topic_filter=topic_filter, max_results=max_results)
        return {"chunks": [r.to_dict() for r in results]}

    def get_lesson(self, lesson_id: str) -> dict:
        return self.lessons.load(lesson_id).to_dict()

    def get_question(self, question_id: str) -> dict:
        return self.db.get_question(question_id)

    def run_python(self, code: str, timeout: int = 30) -> dict:
        return self.executor.run(code, timeout=timeout)

    def get_progress(self, topic: str = None) -> dict:
        return self.db.get_progress(topic)

    def update_progress(self, topic: str, event_type: str, score: float = None, concepts: list[str] = None) -> dict:
        return self.db.update_progress(topic, event_type, score, concepts)

    def get_review_schedule(self) -> dict:
        return self.db.get_review_schedule()

    def get_code_example(self, lesson_id: str, example_name: str = None) -> dict:
        return self.lessons.get_example(lesson_id, example_name)

    def render_diagram(self, diagram_type: str, annotations: dict = None) -> dict:
        return self.diagrams.render(diagram_type, annotations)
```

### 4.4 CLI ↔ Agent Tool Shim Pattern

The agent's Anthropic tool schemas map 1:1 to CLI commands. Each tool's `execute` function is a thin shim that calls the `MLRefresherAPI`:

```python
# agent/tools/content.py
from cli.api import MLRefresherAPI

api = MLRefresherAPI()

async def execute_search_content(tool_input: dict) -> dict:
    """Tool shim: search_content → mlr content search"""
    return api.search_content(
        query=tool_input["query"],
        topic_filter=tool_input.get("topic_filter"),
        max_results=tool_input.get("max_results", 5),
    )

async def execute_get_lesson(tool_input: dict) -> dict:
    """Tool shim: get_lesson → mlr content lesson"""
    return api.get_lesson(lesson_id=tool_input["lesson_id"])

async def execute_get_question(tool_input: dict) -> dict:
    """Tool shim: get_question → mlr content question"""
    return api.get_question(question_id=tool_input["question_id"])
```

Every shim follows the same pattern: extract fields from `tool_input`, call the corresponding `MLRefresherAPI` method, return the dict. No business logic in the shims.

### 4.5 Which Operations Live Where

Not everything is a CLI command. Operations that require the LLM stay in the agent layer:

| Operation | Layer | Why |
|-----------|-------|-----|
| `search_content` | **CLI** (`mlr content search`) | Vector search is deterministic, no LLM needed |
| `get_lesson` | **CLI** (`mlr content lesson`) | File read, no LLM needed |
| `get_question` | **CLI** (`mlr content question`) | Database lookup, no LLM needed |
| `run_python` | **CLI** (`mlr code run`) | Subprocess execution, no LLM needed |
| `get_code_example` | **CLI** (`mlr code example`) | File read, no LLM needed |
| `get_progress` | **CLI** (`mlr progress show`) | Database query, no LLM needed |
| `update_progress` | **CLI** (`mlr progress update`) | Database write, no LLM needed |
| `get_review_schedule` | **CLI** (`mlr progress review`) | Database query + FSRS, no LLM needed |
| `render_diagram` | **CLI** (`mlr diagram show`) | Template lookup, no LLM needed |
| `generate_quiz` | **Agent tool** (no CLI) | Requires LLM to generate questions from content |
| `evaluate_answer` | **Agent tool** (no CLI) | Requires LLM for semantic rubric matching |
| `get_interview_question` | **CLI** (`mlr content next-question`) | Selection logic is deterministic (filter + adaptive difficulty) |

**9 of 11 tools** are backed by CLI commands. The remaining 2 (`generate_quiz`, `evaluate_answer`) are agent-side tools that make side-channel LLM calls — they can't be CLI commands because they require inference.

---

## 5. Agent Tools

The agent's tools are the Anthropic API interface to the CLI. Each tool is defined as an Anthropic tool schema and implemented as a thin shim that calls `MLRefresherAPI`. Two tools (`generate_quiz`, `evaluate_answer`) are agent-side and use side-channel LLM calls instead.

### 5.1 Tool Registry Architecture

```python
@dataclass
class Tool:
    name: str
    description: str
    schema: dict          # Anthropic tool schema (JSON Schema for input)
    execute: Callable     # async (input: dict) -> dict — calls MLRefresherAPI or LLM
    validate: Callable    # (result: dict) -> dict (post-execution validation)
    cli_command: str|None # Corresponding CLI command (None for agent-only tools)
```

All tools are registered in a central `ToolRegistry`. The orchestrator references tools by name when configuring phase-specific `available_tools` lists. Only tools in the current phase's list are sent to the API — the LLM literally cannot call tools it can't see.

### 5.2 Content Retrieval Tools

#### `search_content`
RAG search over all book content (lessons, interview questions, PDFs).

```json
{
  "name": "search_content",
  "description": "Search the ML Refresher book content for information relevant to a query. Returns ranked text chunks with source citations. Use this to find explanations, examples, or background material to ground your responses in the book's content.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language search query describing what content to find."
      },
      "topic_filter": {
        "type": "string",
        "description": "Optional topic ID to restrict search to (e.g., 'attention_mechanisms', 'linear_regression'). Omit to search all content."
      },
      "max_results": {
        "type": "integer",
        "description": "Maximum number of chunks to return. Default 5.",
        "default": 5
      }
    },
    "required": ["query"]
  }
}
```

**Returns:** `{ chunks: [{ text, source_file, topic, lesson_id, relevance_score }] }`

**Implementation:** Embeds the query with `nomic-embed-text-v1.5` (256d Matryoshka), performs hybrid search (BM25 + dense) via LanceDB with optional metadata filter on `topic`, re-ranks top-20 via cross-encoder, returns top-k with source metadata for citation.

#### `get_lesson`
Retrieve a specific lesson's full structured content.

```json
{
  "name": "get_lesson",
  "description": "Retrieve the full content of a specific lesson from the ML Refresher book. Returns the lesson's markdown explanation, code examples, and learning objectives.",
  "input_schema": {
    "type": "object",
    "properties": {
      "lesson_id": {
        "type": "string",
        "description": "Lesson identifier. Format: 'pytorch/01' through 'pytorch/10' for PyTorch lessons, 'interview/01' through 'interview/22' for interview question topics."
      }
    },
    "required": ["lesson_id"]
  }
}
```

**Returns:** `{ title, content_markdown, code_examples: [{ name, code, description }], learning_objectives: [str] }`

**Implementation:** Reads the corresponding directory's README.md + lesson.py/lesson.ipynb, parses into structured output. No vector search — direct file read.

#### `get_question`
Retrieve a specific interview question with its rubric.

```json
{
  "name": "get_question",
  "description": "Retrieve a specific interview question with its expected answer, rubric, and difficulty level. Used for both interview sessions and spaced repetition review.",
  "input_schema": {
    "type": "object",
    "properties": {
      "question_id": {
        "type": "string",
        "description": "Question identifier (e.g., 'attention_q2', 'tokenization_q1')."
      }
    },
    "required": ["question_id"]
  }
}
```

**Returns:** `{ question_id, question_text, expected_answer, rubric: { key_concepts, bonus_concepts, common_mistakes }, difficulty, topic }`

**Implementation:** Looks up the question in the structured rubrics JSON file (`rubrics/questions.json`), which is pre-built from the interview question content during indexing.

### 5.3 Code Execution Tools

#### `run_python`
Execute Python code in a subprocess and return results.

```json
{
  "name": "run_python",
  "description": "Execute Python code and return stdout, stderr, and paths to any saved figures. Use this to demonstrate concepts with live code, run learner experiments, or verify computations. The code runs in the project's uv environment with PyTorch, NumPy, and matplotlib available.",
  "input_schema": {
    "type": "object",
    "properties": {
      "code": {
        "type": "string",
        "description": "Python code to execute. Can use torch, numpy, matplotlib, sklearn, etc."
      },
      "timeout": {
        "type": "integer",
        "description": "Maximum execution time in seconds. Default 30.",
        "default": 30
      }
    },
    "required": ["code"]
  }
}
```

**Returns:** `{ stdout, stderr, exit_code, figures: [str], execution_time_ms }`

**Implementation:** Writes code to a temp file, runs via `uv run python <tempfile>` with `subprocess.run(timeout=...)`. Captures stdout/stderr. If code calls `plt.savefig()`, captures the figure paths. Kills process on timeout.

**Safety:** No sandboxing beyond timeout and resource limits. The user is running code on their own machine in an educational context — the risk profile is the same as running the existing lesson scripts.

#### `get_code_example`
Retrieve a runnable code example from the existing lesson files.

```json
{
  "name": "get_code_example",
  "description": "Retrieve a specific code example from the existing lesson files. Returns the code as a string that can be displayed to the learner or passed to run_python for execution.",
  "input_schema": {
    "type": "object",
    "properties": {
      "lesson_id": {
        "type": "string",
        "description": "Lesson identifier (e.g., 'pytorch/06', 'interview/02')."
      },
      "example_name": {
        "type": "string",
        "description": "Optional name or description of the specific example within the lesson. If omitted, returns the main example."
      }
    },
    "required": ["lesson_id"]
  }
}
```

**Returns:** `{ code, description, source_file, lesson_title }`

**Implementation:** Reads the lesson's .py or .ipynb file. If `example_name` is provided, searches for a matching function/section. Otherwise returns the main demo code.

### 5.4 Assessment Tools

#### `generate_quiz`
Generate quiz questions on a topic, grounded in book content.

```json
{
  "name": "generate_quiz",
  "description": "Generate a set of quiz questions on a specific topic to test the learner's understanding. Questions are grounded in the book's content and calibrated to the specified difficulty level.",
  "input_schema": {
    "type": "object",
    "properties": {
      "topic": {
        "type": "string",
        "description": "Topic to generate questions about (e.g., 'attention_mechanisms', 'gradients')."
      },
      "difficulty": {
        "type": "string",
        "enum": ["novice", "intermediate", "advanced"],
        "description": "Difficulty level for the questions."
      },
      "count": {
        "type": "integer",
        "description": "Number of questions to generate. Default 3.",
        "default": 3,
        "minimum": 1,
        "maximum": 10
      }
    },
    "required": ["topic", "difficulty"]
  }
}
```

**Returns:** `{ questions: [{ id, question_text, answer_key, concepts_tested: [str] }] }`

**Implementation:** This is a **tool that itself calls the LLM** — it makes a separate Anthropic API call with the topic's retrieved content as context, asking for structured quiz generation. The quiz output is validated (must have questions, must have answer keys) before returning. This is NOT the same LLM turn as the conversation — it's a side-channel call with a quiz-generation-specific system prompt.

#### `evaluate_answer`
Score a user's answer against a rubric.

```json
{
  "name": "evaluate_answer",
  "description": "Evaluate the learner's answer to a question against the expected answer and rubric. Returns a structured score with concept coverage analysis, identified gaps, and constructive feedback.",
  "input_schema": {
    "type": "object",
    "properties": {
      "question_id": {
        "type": "string",
        "description": "The ID of the question being evaluated."
      },
      "user_answer": {
        "type": "string",
        "description": "The learner's answer text to evaluate."
      }
    },
    "required": ["question_id", "user_answer"]
  }
}
```

**Returns:**
```json
{
  "score": 75,
  "concepts_covered": ["parallel attention heads", "separate projections"],
  "concepts_missed": ["final linear projection after concatenation"],
  "common_mistakes_triggered": [],
  "feedback": "Strong explanation of the parallel computation. You missed the final projection step — after concatenating head outputs, there's a linear transformation W_O that maps back to d_model dimensions.",
  "difficulty_recommendation": "stay"
}
```

**Implementation:** Another side-channel LLM call. Sends the user's answer + the rubric (from `questions.json`) to the LLM with a structured output prompt. The rubric provides the ground truth; the LLM does semantic matching (not string matching) to determine which concepts were covered.

**Validation:** Score must be 0-100. `concepts_covered` and `concepts_missed` must be subsets of the rubric's concept lists. `difficulty_recommendation` must be one of `escalate | stay | de-escalate`.

#### `get_interview_question`
Select the next interview question based on adaptive difficulty.

```json
{
  "name": "get_interview_question",
  "description": "Select the next interview question for a mock interview session. Uses the learner's current performance in this session to adaptively select difficulty. Draws from the existing 50+ question bank.",
  "input_schema": {
    "type": "object",
    "properties": {
      "topic": {
        "type": "string",
        "description": "Topic to draw questions from. If omitted, selects across all topics."
      },
      "difficulty": {
        "type": "string",
        "enum": ["novice", "intermediate", "advanced"],
        "description": "Explicit difficulty override. If omitted, adapts based on session performance."
      },
      "exclude_ids": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Question IDs already asked in this session (to avoid repeats)."
      }
    }
  }
}
```

**Returns:** `{ question_id, question_text, difficulty, topic, suggested_followups: [str], rubric }`

**Implementation:** Queries the structured question bank. If `difficulty` is omitted, uses the session's running average score to select: >= 80% → escalate, 50-80% → same level, < 50% → de-escalate. Filters out `exclude_ids`. Returns the question with its rubric (for later evaluation) and suggested follow-up probes.

### 5.5 State Management Tools

#### `get_progress`
Get the user's learning state.

```json
{
  "name": "get_progress",
  "description": "Retrieve the learner's current progress and knowledge state. Returns per-topic levels, recent scores, weak areas, and overall statistics. Use this at the start of a session to calibrate difficulty and at any point to understand where the learner stands.",
  "input_schema": {
    "type": "object",
    "properties": {
      "topic": {
        "type": "string",
        "description": "Specific topic to get detailed progress for. If omitted, returns overall summary."
      }
    }
  }
}
```

**Returns (topic-specific):** `{ topic, level, total_interactions, recent_scores: [float], weak_subtopics: [str], last_session_date }`

**Returns (overall):** `{ topics_started, topics_at_novice, topics_at_intermediate, topics_at_advanced, overall_score_avg, weakest_topics: [str], strongest_topics: [str] }`

**Implementation:** SQLite queries against the `topics`, `learning_events`, and `fsrs_cards` tables.

#### `update_progress`
Record a learning event.

```json
{
  "name": "update_progress",
  "description": "Record a learning event (quiz result, lesson completion, interview score, review). Updates the learner's knowledge state and FSRS spaced repetition cards.",
  "input_schema": {
    "type": "object",
    "properties": {
      "topic": {
        "type": "string",
        "description": "Topic ID for this event."
      },
      "event_type": {
        "type": "string",
        "enum": ["quiz", "lesson_complete", "interview", "review"],
        "description": "Type of learning event."
      },
      "score": {
        "type": "number",
        "description": "Score for this event (0.0 to 1.0). Required for quiz, interview, and review events.",
        "minimum": 0.0,
        "maximum": 1.0
      },
      "concepts_tested": {
        "type": "array",
        "items": { "type": "string" },
        "description": "List of specific concepts tested in this event (for FSRS card updates)."
      }
    },
    "required": ["topic", "event_type"]
  }
}
```

**Returns:** `{ updated_level, total_interactions, level_changed: bool }`

**Implementation:** Inserts a row into `learning_events`. If `concepts_tested` is provided, updates the corresponding FSRS cards via `py-fsrs`. If the running average of recent scores crosses a threshold, promotes/demotes the topic's level (novice → intermediate at avg >= 0.7, intermediate → advanced at avg >= 0.85).

#### `get_review_schedule`
Check what's due for spaced repetition review.

```json
{
  "name": "get_review_schedule",
  "description": "Check which concepts are due for spaced repetition review. Returns concepts sorted by urgency (most overdue first). Use this at the start of a Teacher session to begin with a warm-up review question.",
  "input_schema": {
    "type": "object",
    "properties": {}
  }
}
```

**Returns:** `{ due_items: [{ topic, concept, days_overdue, last_score, card_state }], total_due }`

**Implementation:** Queries `fsrs_cards` where `due_date <= today`, sorted by `due_date` ascending (most overdue first). Includes the card's last score and state so the agent can select an appropriate review question.

### 5.6 Presentation Tools

#### `render_diagram`
Render an ASCII/Unicode architecture diagram.

```json
{
  "name": "render_diagram",
  "description": "Render a pre-built ASCII architecture diagram for common ML concepts. Use this to visually illustrate architectures during explanations.",
  "input_schema": {
    "type": "object",
    "properties": {
      "diagram_type": {
        "type": "string",
        "enum": [
          "transformer_full",
          "transformer_encoder",
          "transformer_decoder",
          "self_attention",
          "multi_head_attention",
          "encoder_decoder",
          "feed_forward",
          "positional_encoding",
          "layer_norm",
          "embedding_lookup",
          "cnn_architecture",
          "rnn_unrolled",
          "lstm_cell",
          "autoencoder",
          "gan_architecture"
        ],
        "description": "Type of architecture diagram to render."
      },
      "annotations": {
        "type": "object",
        "description": "Optional key-value pairs to annotate on the diagram (e.g., {'d_model': '512', 'num_heads': '8'})."
      }
    },
    "required": ["diagram_type"]
  }
}
```

**Returns:** `{ diagram: str, description: str }`

**Implementation:** Pre-built ASCII art templates stored as string constants. If `annotations` are provided, substitutes values into labeled placeholders in the template. This is NOT LLM-generated — it's deterministic, fast, and always correct.

### 5.7 Tool Summary by Phase

**Teacher phases:**

| Phase | Available Tools | Forced First Call |
|-------|----------------|-------------------|
| WARMUP | `get_review_schedule`, `get_question`, `evaluate_answer` | `get_review_schedule` |
| INTRODUCE | `search_content`, `get_lesson`, `render_diagram` | `search_content` |
| EXPLORE | `search_content`, `run_python`, `get_code_example`, `render_diagram` | — (LLM chooses) |
| PRACTICE | `generate_quiz`, `evaluate_answer`, `run_python`, `search_content` | `generate_quiz` |
| WRAPUP | `update_progress` | `update_progress` |

**Interviewer phases:**

| Phase | Available Tools | Forced First Call |
|-------|----------------|-------------------|
| SETUP | `get_progress` | `get_progress` |
| QUESTION | `get_interview_question`, `search_content` | `get_interview_question` |
| FOLLOWUP | `search_content` | — (LLM chooses) |
| EVALUATE | `evaluate_answer`, `search_content` | `evaluate_answer` |
| DEBRIEF | `update_progress`, `get_progress` | `update_progress` |

---

## 6. Teacher Persona

### 6.1 Pedagogical Approach

The Teacher uses the **Socratic method** — it guides the learner to discover answers rather than lecturing at them.

**Questioning hierarchy** (from [Socratic AI research](https://princeton-nlp.github.io/SocraticAI/)):
1. **Clarification**: "What do you mean by...?"
2. **Assumption probing**: "What assumption are you making about the softmax here?"
3. **Evidence seeking**: "Can you trace through the tensor dimensions to verify?"
4. **Perspective shifting**: "How would this differ in an encoder vs decoder?"
5. **Consequence exploring**: "What happens if we don't scale by sqrt(d_k)?"

The agent should NOT give direct answers unless the learner is stuck after 2-3 guided questions.

### 6.2 Adaptive Difficulty

Track three levels per topic:

| Level | Behavior |
|-------|----------|
| **Novice** | Explain with analogies and simple examples. More scaffolding. Shorter code demos. |
| **Intermediate** | Ask "why" questions. Expect learner to connect concepts across topics. |
| **Advanced** | Present edge cases, failure modes, tradeoffs. Challenge assumptions. Discuss paper-level details. |

Level is determined by quiz performance and conversation quality. The agent uses the `get_progress` tool at session start to calibrate.

### 6.3 Guided Session Flow

A Teacher session follows this structure:

```
1. WARM-UP (2 min)
   └─ Spaced repetition: review question from a previously learned topic
      (Agent calls get_review_schedule → get_question → evaluate_answer → update_progress)

2. CONCEPT INTRODUCTION (5 min)
   └─ Present today's topic, grounded in book content
      (Agent calls search_content → get_lesson)
   └─ Provide intuition before formalism

3. INTERACTIVE EXPLORATION (10-15 min)
   └─ Socratic dialogue about the concept
   └─ Code demonstrations (Agent calls get_code_example → run_python)
   └─ "What if" experiments ("What if we change the learning rate?")
   └─ Architecture diagrams (Agent calls render_diagram)

4. PRACTICE (5-10 min)
   └─ Generate and administer a quiz (Agent calls generate_quiz)
   └─ Evaluate answers (Agent calls evaluate_answer)
   └─ Discuss mistakes

5. WRAP-UP (2 min)
   └─ Summarize key takeaways
   └─ Preview next topic
   └─ Update learning state (Agent calls update_progress)
```

The agent drives this flow autonomously — the user doesn't have to know about the structure.

### 6.4 Content Grounding

All Teacher explanations must be grounded in the existing book content via RAG. The system prompt instructs:

> "Base your explanations on retrieved content from the ML Refresher book. Cite your sources (e.g., 'As covered in Lesson 6: Linear Regression...'). If the retrieved content doesn't cover a topic, acknowledge this rather than generating ungrounded explanations."

---

## 7. Interviewer Persona

### 7.1 Interview Session Flow

```
1. SETUP
   └─ Select topic (user chooses or random)
   └─ Select difficulty (or adaptive based on get_progress)
   └─ Set interview parameters (# questions, time limit optional)

2. QUESTION LOOP (repeat for each question)
   ├─ Ask question (from the existing 50+ question bank)
   │   (Agent calls get_interview_question)
   ├─ User answers
   ├─ Follow-up probe (1-2 based on answer quality):
   │   ├─ Clarification: "Can you elaborate on..."
   │   ├─ Extension: "How would this change for..."
   │   ├─ Challenge: "What's the computational complexity?"
   │   └─ Application: "How would you apply this to..."
   ├─ Evaluate answer against rubric
   │   (Agent calls evaluate_answer)
   └─ Reveal: model answer, concept gaps, score

3. DEBRIEF
   └─ Overall score and breakdown by topic
   └─ Identified weak areas
   └─ Recommended study topics
   └─ Update learning state (Agent calls update_progress)
```

### 7.2 Rubric-Based Evaluation

Each interview question has a structured rubric:

```json
{
  "question_id": "attention_02",
  "question": "Explain how multi-head attention works and why it's useful.",
  "difficulty": "intermediate",
  "rubric": {
    "key_concepts": [
      "Multiple parallel attention heads with separate Q/K/V projections",
      "Each head learns different representation subspaces",
      "Outputs concatenated and linearly projected",
      "Enables attending to different positions/features simultaneously"
    ],
    "bonus_concepts": [
      "Relationship between d_model, d_k, and num_heads",
      "Computational cost equivalence to single large-head attention",
      "Connection to ensemble methods"
    ],
    "common_mistakes": [
      "Confusing multi-head with multi-layer attention",
      "Omitting the final linear projection after concatenation",
      "Not explaining WHY multiple heads help (just stating they do)"
    ]
  }
}
```

The agent evaluates concept coverage, not exact wording. Scoring: percentage of key concepts mentioned, bonus for bonus concepts, flags for common mistakes.

### 7.3 Adaptive Difficulty

Within a session:
- Score >= 80%: escalate to harder questions or deeper follow-ups
- Score 50-80%: stay at current level, probe weak sub-areas
- Score < 50%: step back, offer hints, suggest the user switch to Teacher mode for that topic

---

## 8. RAG System

The RAG system is a deliberate over-investment relative to the corpus size (~500-1000 chunks). This is intentional — it serves as a learning exercise in modern retrieval engineering alongside its functional purpose.

### 8.1 Embedding Model: `nomic-embed-text-v1.5`

**Decision: Replace `all-MiniLM-L6-v2` with `nomic-embed-text-v1.5`.**

| Property | all-MiniLM-L6-v2 | nomic-embed-text-v1.5 |
|----------|-------------------|------------------------|
| Parameters | 22.7M | 137M |
| Model size | ~80MB | ~274MB |
| Max tokens | **128** | **8192** |
| Dimensions | 384 (fixed) | 768 (Matryoshka: 64-768) |
| MTEB average | 56.26 | 62.28 |
| Retrieval NDCG@10 | 41.66 | **53.01** |
| Instruction prefixes | No | Yes (`search_query:` / `search_document:`) |
| Matryoshka support | No | Yes (768, 512, 256, 128, 64) |

**Why nomic wins for this project:**

1. **8192-token context**: Critical. Our content has code blocks and technical explanations that frequently exceed MiniLM's 128-token limit. A function with a docstring can easily be 200+ tokens — MiniLM silently truncates it, losing information. Nomic embeds the full chunk.

2. **+11.35 points on retrieval NDCG@10** (53.01 vs 41.66): This is a massive gap. On BEIR benchmarks, nomic retrieves the correct document in the top 10 results ~27% more often than MiniLM.

3. **Matryoshka dimensions**: We use **256 dimensions** (not the full 768). Research shows 256d retains 98% of retrieval quality while giving 3x storage savings and faster similarity computation. The quality/dimension tradeoff for nomic-embed-text-v1.5:

```
Dimension    MTEB Score    % of Full     Storage vs 768d
768          62.28         100.0%        100%
512          61.96          99.5%         67%
256          61.04          98.0%         33%    ◄── our choice
128          59.34          95.3%         17%
64           56.10          90.1%          8%
```

4. **Instruction prefixes improve retrieval**: Nomic uses differentiated prefixes (`search_query:` for queries, `search_document:` for documents) that break the symmetry in contrastive training. This is not optional — the model was trained with these prefixes and performs worse without them.

**Embedding code:**

```python
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
MATRYOSHKA_DIM = 256

def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed document chunks with search_document prefix and Matryoshka truncation."""
    prefixed = [f"search_document: {t}" for t in texts]
    embeddings = model.encode(prefixed, convert_to_tensor=True)
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = embeddings[:, :MATRYOSHKA_DIM]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def embed_query(query: str) -> np.ndarray:
    """Embed a search query with search_query prefix."""
    prefixed = f"search_query: {query}"
    embedding = model.encode([prefixed], convert_to_tensor=True)
    embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
    embedding = embedding[:, :MATRYOSHKA_DIM]
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.cpu().numpy()
```

**Quantization**: We apply **int8 scalar quantization** to stored embeddings for 4x additional storage savings. Combined with 256d Matryoshka, this gives **12x total storage reduction** (3x dimension × 4x quantization) over naive float32-768d, with ~95-96% quality retention. For ~1000 chunks, the entire index fits in ~100KB instead of ~3MB — trivial either way at this scale, but the technique is worth implementing for the learning exercise.

### 8.2 Vector Store: LanceDB

**Decision: Replace ChromaDB with LanceDB.**

| Feature | ChromaDB | LanceDB |
|---------|----------|---------|
| Native hybrid search | Yes (Nov 2025+, sparse vectors) | **Yes (built-in tantivy BM25 + RRF)** |
| Hybrid search maturity | New, sparse vector API | **Mature, single-line API** |
| Metadata filtering | Yes (`where` clauses) | **Yes (SQL-style via DataFusion)** |
| Storage format | SQLite + hnswlib | **Lance columnar (Rust)** |
| Parent-child retrieval | Manual (metadata references) | **Better relational support** |
| Setup | `pip install chromadb` | `pip install lancedb` |
| Embedded mode | Yes | **Yes (serverless-first, no process)** |
| Python API | Good (NumPy-like) | **Excellent (pandas/Arrow integration)** |

**Why LanceDB wins:**

1. **Native hybrid search in one line**: `table.search(query, query_type="hybrid")`. Uses tantivy (Rust-based BM25) internally with Reciprocal Rank Fusion. No need to wire up a separate BM25 index manually.

2. **SQL-style filtering**: `table.search(vector).where("topic = 'attention' AND has_code = true")` via DataFusion. More expressive than ChromaDB's `where` dict.

3. **Parent-child support**: Relational patterns are natural in LanceDB's Arrow-based schema. Store parent and child records in the same table with cascading references.

At our corpus size (<1000 chunks), both databases return results in <10ms. The choice is about API ergonomics and native hybrid search, not performance.

### 8.3 Retrieval Pipeline: Hybrid Search + Re-ranking

The retrieval pipeline has three stages. Research shows this stack achieves ~94% recall / ~78% precision — a dramatic improvement over naive cosine similarity (~72% recall / ~48% precision).

```
Query
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: Hybrid Retrieval (BM25 + Dense)              ~15ms │
│                                                              │
│  ┌─────────────────┐    ┌──────────────────┐                │
│  │ BM25 (tantivy)  │    │ Dense (nomic     │                │
│  │ keyword match   │    │ 256d cosine sim) │                │
│  │ top-20          │    │ top-20           │                │
│  └────────┬────────┘    └────────┬─────────┘                │
│           │                      │                           │
│           └──────────┬───────────┘                           │
│                      ▼                                       │
│              ┌───────────────┐                               │
│              │ RRF Fusion    │  Reciprocal Rank Fusion       │
│              │ top-20 merged │  score = Σ 1/(k + rank_i)    │
│              └───────┬───────┘  k = 60 (standard)           │
│                      │                                       │
└──────────────────────┼───────────────────────────────────────┘
                       │ 20 candidates
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 2: Metadata Filtering                            ~0ms │
│                                                              │
│  Apply phase-contextual filters:                            │
│  - topic = current_topic (if in a specific lesson)          │
│  - source_type = "pytorch_lesson" | "interview_questions"   │
│  - has_code = true (if agent asked for code examples)       │
│                                                              │
│  Pre-filtering (before vector search) when topic is known.  │
│  Post-filtering when topic is unknown (exploratory queries).│
│                                                              │
└──────────────────────┼───────────────────────────────────────┘
                       │ 10-20 filtered candidates
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 3: Cross-Encoder Re-ranking                    ~45ms  │
│                                                              │
│  Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (ONNX)       │
│  - 22M params, ~45ms for 20 candidates on CPU              │
│  - Encodes (query, document) pairs jointly                  │
│  - Much more precise than bi-encoder cosine similarity      │
│  - ONNX backend: 2-3x faster than PyTorch on CPU           │
│                                                              │
│  Input: 20 candidates from Stage 1+2                        │
│  Output: top-5 re-ranked by cross-encoder score             │
│                                                              │
└──────────────────────┼───────────────────────────────────────┘
                       │ top-5 results
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 4: Parent Expansion                              ~1ms │
│                                                              │
│  For each child chunk in top-5:                             │
│  - Look up parent_id in metadata                            │
│  - Fetch the parent chunk (full question or lesson section) │
│  - Return parent content to the LLM for context            │
│  - Keep child chunk source info for citation                │
│                                                              │
│  "Retrieve small, return big"                               │
└──────────────────────┼───────────────────────────────────────┘
                       │
                       ▼
              Final context for LLM
              (parent chunks + source citations)
```

**Total retrieval latency: ~60ms on modern CPU**. This is well under the threshold for interactive use — the LLM API call dominates at 500-2000ms.

**Why each stage matters:**

| Stage | What it adds | Impact |
|-------|-------------|--------|
| **BM25 (sparse)** | Catches exact keyword matches that vector search misses ("LoRA", "KL divergence", "HNSW") | +22 points recall over dense-only |
| **Dense (nomic)** | Catches semantic matches ("regularization technique" → "dropout") | Baseline semantic retrieval |
| **RRF fusion** | Combines both without score normalization | Robust, no hyperparameter tuning |
| **Metadata filter** | Prevents cross-topic contamination (attention query doesn't pull loss function chunks) | Significant precision gain, ~0ms cost |
| **Cross-encoder** | Jointly models query-document interaction; catches subtle relevance signals bi-encoders miss | +17 points precision |
| **Parent expansion** | Returns full context (entire Q&A block) instead of just the matched fragment | LLM gets enough context to generate good answers |

**Re-ranker implementation:**

```python
from sentence_transformers import CrossEncoder

# Load with ONNX backend for 2-3x CPU speedup
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    backend="onnx",
)

def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Re-rank candidates using cross-encoder."""
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked[:top_k]]
```

### 8.4 Content Chunking Strategy

**Core principle: Structure-aware splitting, not token-count splitting.**

Our content has explicit markdown headers (`## Q{n}:`, `### Section`, `---`). These are better semantic boundaries than any ML-based topic detection. Research confirms: a 2026 benchmark of 7 chunking strategies found recursive splitting at structural boundaries achieved 69% accuracy vs 54% for semantic (embedding-based) chunking, largely because semantic chunking produced fragments averaging only 43 tokens.

**Rules:**

1. **Never use fixed-size token splitting** for markdown or Python files. Use structural boundaries.
2. **Never split mid-code-block**. Code blocks (fenced with ```) are atomic units.
3. **Never use overlap**. Our structural boundaries produce self-contained chunks. Overlap is a patch for bad boundary detection.
4. **Keep prose + adjacent code together**. A code block with its surrounding explanation is a single chunk.

#### 8.4.1 Interview Questions Chunking

```
Parent chunk: Full question + answer block (everything between ## Q{n} headers)
Child chunks: Individual ### sub-sections within each question

Example for interview_questions/02_attention_mechanisms/README.md:

Parent: "Q2: How does the attention mechanism work?"
  └─ Child: "The Core Idea" (prose, ~200 tokens)
  └─ Child: "The Attention Formula" (prose + math, ~150 tokens)
  └─ Child: "Key Components" (prose + code block, ~300 tokens)
  └─ Child: "Visual Example" (prose + ASCII diagram, ~250 tokens)
```

Retrieval searches child chunks (precise matching). Parent chunks are returned to the LLM (broad context).

#### 8.4.2 PyTorch Lessons Chunking

```
Parent chunk: Full section (e.g., "Key Concepts", "Code Walkthrough")
Child chunks: Individual concepts or code examples within each section

Example for pytorch_refresher/05_tensor_gradients/README.md:

Parent: "Key Concepts"
  └─ Child: "Creating Gradient-Tracking Tensors" (prose + code, ~250 tokens)
  └─ Child: "The Computational Graph" (prose, ~200 tokens)
  └─ Child: "Computing Gradients" (prose + code + output, ~350 tokens)

Parent: "Code Walkthrough"
  └─ Child: "gradient_demo()" function (code + docstring, ~300 tokens)
```

#### 8.4.3 Python Files Chunking

For standalone `.py` and `.ipynb` files:

- Split at function/class boundaries using AST-aware splitting (`RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` or `ASTChunk`)
- Each function + its docstring/comments = one chunk
- Attach metadata linking back to the parent README section

#### 8.4.4 PDF Sources Chunking

For `docs/*.pdf`:
- Convert to structured text first using Docling or `pymupdf4llm`
- Apply the same markdown-header-based splitting as interview questions
- Largest chunks; these are reference material, not primary retrieval targets

### 8.5 Contextual Chunk Enrichment

Before embedding, each chunk gets a **template-based context prefix** prepended. This situates the chunk within the document hierarchy, resolving ambiguity in retrieval.

**Implementation (no LLM calls — deterministic, free):**

```python
def add_context_prefix(chunk_text: str, metadata: dict) -> str:
    """Prepend structural context to chunk before embedding."""
    if metadata["source_type"] == "interview_questions":
        prefix = (
            f"From LLM Interview Questions, "
            f"Category: {metadata['category'].replace('_', ' ').title()}, "
            f"{metadata['question_id']}: {metadata['question_text']}. "
            f"Section: {metadata.get('subsection', 'Overview')}."
        )
    elif metadata["source_type"] == "pytorch_lesson":
        prefix = (
            f"From PyTorch Refresher, "
            f"Lesson {metadata['lesson_number']}: {metadata['lesson_title']}. "
            f"Section: {metadata['section']}."
        )
    elif metadata["source_type"] == "python_file":
        prefix = (
            f"Code from {metadata['file_path']}, "
            f"function: {metadata.get('function_name', 'module-level')}."
        )
    return f"{prefix}\n\n{chunk_text}"
```

**Example before/after:**

Before: `"LoRA works by freezing the original weights and adding low-rank decomposition matrices A and B such that the weight update is W + BA."`

After: `"From LLM Interview Questions, Category: Fine Tuning Methods, Q45: What is LoRA?. Section: How It Works.\n\nLoRA works by freezing the original weights and adding low-rank decomposition matrices A and B such that the weight update is W + BA."`

This gets 80% of the benefit of Anthropic's full contextual retrieval approach (which uses an LLM call per chunk) at zero cost. Research shows contextual retrieval reduces retrieval failures by 35-67%. Template-based context captures the structural signal, which is the primary source of ambiguity in hierarchical educational content.

**Optional upgrade**: If template-based context proves insufficient, run LLM-generated context enrichment as a one-time preprocessing step. Cost for our corpus: ~$0.50 with Claude Haiku. Decision deferred to implementation.

### 8.6 Metadata Schema

Every chunk carries structured metadata for filtering and citation:

```python
# Interview question chunks
{
    "source_type": "interview_questions",
    "category": "attention_mechanisms",
    "category_number": 2,
    "question_id": "Q2",
    "question_text": "How does the attention mechanism work?",
    "section": "Key Components",
    "content_type": "explanation",       # explanation | formula | code | comparison | definition
    "has_code": True,
    "difficulty": "intermediate",
    "parent_id": "interview_02_q2",
    "level": "child",                    # parent | child
    "file_path": "interview_questions/02_attention_mechanisms/README.md",
}

# PyTorch lesson chunks
{
    "source_type": "pytorch_lesson",
    "lesson_number": 5,
    "lesson_title": "Tensor Gradients",
    "section": "Key Concepts",
    "subsection": "Computing Gradients",
    "content_type": "explanation",
    "has_code": True,
    "parent_id": "pytorch_05_key_concepts",
    "level": "child",
    "file_path": "pytorch_refresher/05_tensor_gradients/README.md",
}

# Python file chunks
{
    "source_type": "python_file",
    "function_name": "gradient_demo",
    "parent_readme": "pytorch_refresher/05_tensor_gradients/README.md",
    "parent_id": "pytorch_05_code_walkthrough",
    "level": "child",
    "file_path": "pytorch_refresher/05_tensor_gradients/lesson.py",
}
```

**Metadata is used at retrieval time** for pre-filtering:
- Agent is in the "Attention Mechanisms" lesson → filter to `category = "attention_mechanisms"`
- Agent asks for code examples → filter to `has_code = true`
- Agent needs a specific question → filter to `question_id = "Q2"`

### 8.7 Indexing Pipeline

```
mlr index build [--force]
```

**Pipeline stages:**

```
1. WALK content directories
   ├── interview_questions/**/*.md
   ├── pytorch_refresher/**/*.md
   ├── pytorch_refresher/**/*.py
   ├── pytorch_refresher/**/*.ipynb
   └── docs/*.pdf

2. PARSE each file type
   ├── Markdown: split on ## and ### headers, keep code blocks atomic
   ├── Python: AST-aware function/class boundary splitting
   ├── Notebooks: one chunk per cell, attach preceding markdown as context
   └── PDFs: convert to markdown first (Docling/pymupdf4llm), then split

3. BUILD parent-child relationships
   ├── Parents: full question blocks, full lesson sections
   └── Children: sub-sections, individual code examples

4. ENRICH with context prefix
   └── Prepend template-based structural context to each chunk

5. EMBED with nomic-embed-text-v1.5
   ├── search_document: prefix for all chunks
   ├── Matryoshka truncation to 256 dimensions
   ├── Layer norm → truncate → L2 normalize
   └── Optional: int8 scalar quantization

6. STORE in LanceDB
   ├── Chunks table: id, text, vector, parent_id, level, all metadata fields
   ├── FTS index on text column (for BM25)
   └── Metadata indexes on category, source_type, question_id

7. BUILD structured index
   └── Extract questions + rubrics → rubrics/questions.json (for Interviewer)

8. VERIFY
   └── mlr index status: report chunk count, parent/child ratio, index size
```

**Expected output for our corpus:**
- ~300-500 child chunks (primary retrieval targets)
- ~80-120 parent chunks (context for LLM)
- Index size: ~200KB (256d × int8 × ~500 chunks)
- Build time: ~30-60 seconds (embedding dominates; CPU-bound)

### 8.8 Retrieval Strategy Summary

| Technique | Implementation | Impact | Latency | Priority |
|-----------|---------------|--------|---------|----------|
| **Hybrid search (BM25 + dense)** | LanceDB native (`query_type="hybrid"`) | +22 pts recall | +15ms | P0 — highest ROI |
| **Metadata pre-filtering** | LanceDB `.where()` clauses | Major precision gain | ~0ms | P0 — nearly free |
| **Cross-encoder re-ranking** | `ms-marco-MiniLM-L-6-v2` (ONNX) | +17 pts precision | +45ms | P0 — high impact |
| **Parent-child retrieval** | Metadata `parent_id` + second lookup | LLM gets full context | +1ms | P0 — architectural |
| **Template context prefixes** | Deterministic string prepend | ~35% fewer retrieval failures | 0ms (index time) | P0 — free |
| **Nomic embed (256d Matryoshka)** | `nomic-embed-text-v1.5` truncated | +11 pts NDCG vs MiniLM | Same | P0 — model swap |
| **Int8 quantization** | `sentence_transformers.quantize_embeddings` | 4x storage savings | Negligible | P1 — learning exercise |
| **LLM contextual enrichment** | One-time Claude Haiku call per chunk | Additional precision | ~$0.50 one-time | P2 — if needed |
| **Query decomposition** | Agent decomposes → multiple searches | Better for complex Qs | +LLM call | P2 — if needed |
| **HyDE** | Generate hypothetical answer → embed | Better for vague queries | +LLM call | P3 — experimental |

---

## 9. Knowledge State & Spaced Repetition

### 9.1 User Progress Schema

```sql
CREATE TABLE topics (
    id TEXT PRIMARY KEY,           -- e.g., "attention_mechanisms"
    display_name TEXT,
    category TEXT,                 -- "pytorch" | "interview"
    level TEXT DEFAULT 'novice',   -- novice | intermediate | advanced
    total_interactions INTEGER DEFAULT 0
);

CREATE TABLE learning_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id TEXT REFERENCES topics(id),
    event_type TEXT,               -- quiz | lesson_complete | interview | review
    score REAL,                    -- 0.0 to 1.0
    timestamp TEXT,
    metadata TEXT                  -- JSON blob for event-specific data
);

CREATE TABLE fsrs_cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id TEXT REFERENCES topics(id),
    concept TEXT,                  -- specific concept within topic
    stability REAL,
    difficulty REAL,
    due_date TEXT,
    last_review TEXT,
    reps INTEGER DEFAULT 0,
    lapses INTEGER DEFAULT 0,
    state TEXT DEFAULT 'new'       -- new | learning | review | relearning
);
```

### 9.2 FSRS Integration

Use `py-fsrs` to schedule concept reviews:
- After each quiz/evaluation, create or update FSRS cards for the tested concepts
- At session start (Teacher mode), check for due reviews and begin with a warm-up question
- The `get_review_schedule` tool queries FSRS for items due today, sorted by overdue-ness

---

## 10. TUI Design

### 10.1 Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  ML Refresher                              [Teacher] [Interview] │
├──────────────┬──────────────────────────────────────────────────┤
│              │                                                  │
│  TOPICS      │  🎓 Teacher: Let's explore attention mechanisms. │
│  ──────      │                                                  │
│  ▸ Tensors   │  Before we dive in, quick review: last session   │
│  ▸ Reshaping │  we covered embeddings. Can you explain what     │
│  ▸ Indexing  │  an embedding does and why we use them instead   │
│  ▸ Math Ops  │  of one-hot vectors?                             │
│  ▸ Gradients │                                                  │
│  ▸ Lin. Reg. │  ──────────────────────────────────────────────  │
│    ...       │                                                  │
│              │  You: Embeddings are dense vector representations │
│  PROGRESS    │  that capture semantic meaning in a lower...      │
│  ──────      │                                                  │
│  Overall: 34%│  🎓 Teacher: Good start! You mentioned "semantic │
│  Attention ●●○│  meaning" — can you be more specific? What kind │
│  Tokeniz. ●○○│  of relationships do embeddings capture that     │
│  Loss Fns ○○○│  one-hot vectors miss?                           │
│              │                                                  │
│              │  ──────────────────────────────────────────────  │
│              │  > Type your response...                    [⏎]  │
├──────────────┴──────────────────────────────────────────────────┤
│  Topic: Attention Mechanisms  │  Level: Intermediate  │  ↑↓ Nav │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Key UI Components

| Component | Textual Widget | Behavior |
|-----------|---------------|----------|
| Chat messages | Custom widget extending `Markdown` | Streaming via `MarkdownStream` (Textual v4). Teacher messages use accent color, user messages use neutral. |
| Input box | `Input` | Multi-line text entry. Submit on Enter (Shift+Enter for newline). |
| Sidebar - Topics | `ListView` with `ListItem`s | Hierarchical topic list. Click to navigate. Shows completion indicators. |
| Sidebar - Progress | Custom `ProgressBar` widgets | Per-topic progress (novice/intermediate/advanced as filled dots). |
| Mode switcher | `TabbedContent` or header buttons | Switch between Teacher and Interviewer. Confirms before discarding active session. |
| Status bar | `Footer` | Current topic, difficulty level, keyboard shortcuts. |

### 10.3 Key Interactions

- **Ctrl+T**: Switch to Teacher mode
- **Ctrl+I**: Switch to Interviewer mode
- **Ctrl+N**: New session (select topic)
- **Ctrl+P**: View progress dashboard
- **Ctrl+Q**: Quit
- **Up/Down**: Scroll chat history
- **Tab**: Focus sidebar / chat input

---

## 11. Project Structure

```
ml-refresher/
│
├── cli/                            # ═══ THE CLI (repository operations layer) ═══
│   ├── __init__.py
│   ├── __main__.py                 # Entry point: `uv run python -m cli` / `mlr`
│   ├── api.py                      # MLRefresherAPI: programmatic interface
│   │                               #   (same operations as CLI, no Click dependency,
│   │                               #    called by agent tool shims in-process)
│   ├── commands/                   # Click command groups
│   │   ├── __init__.py
│   │   ├── content.py              # mlr content {search, lesson, question, topics, questions}
│   │   ├── code.py                 # mlr code {run, example}
│   │   ├── progress.py             # mlr progress {show, update, review}
│   │   ├── diagram.py              # mlr diagram {show, list}
│   │   └── index.py                # mlr index {build, status}
│   ├── services/                   # Business logic (shared by CLI commands + API)
│   │   ├── __init__.py
│   │   ├── retriever.py            # LanceDB hybrid search + cross-encoder re-ranking
│   │   ├── lessons.py              # Lesson file loading + parsing
│   │   ├── questions.py            # Question bank + rubric management
│   │   ├── executor.py             # Python code execution (subprocess + timeout)
│   │   ├── diagrams.py             # ASCII diagram templates + rendering
│   │   └── indexer.py              # Content chunking + embedding pipeline
│   ├── state/                      # Persistence layer
│   │   ├── __init__.py
│   │   ├── db.py                   # SQLite schema + migrations + queries
│   │   ├── progress.py             # Learning state management
│   │   └── fsrs.py                 # FSRS integration for spaced repetition
│   ├── chunking.py                 # Hierarchical content chunking strategies
│   └── rubrics/
│       └── questions.json          # Structured rubrics (generated by indexer)
│
├── tui/                            # ═══ THE TUI (presentation layer) ═══
│   ├── __init__.py
│   ├── __main__.py                 # Entry point: `uv run python -m tui`
│   ├── app.py                      # Textual App definition
│   ├── screens/
│   │   ├── __init__.py
│   │   ├── chat.py                 # Main chat screen (Teacher & Interviewer)
│   │   ├── welcome.py              # Welcome / mode selection screen
│   │   └── progress.py             # Progress dashboard screen
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── chat_message.py         # Streaming markdown message widget
│   │   ├── sidebar.py              # Topic nav + progress sidebar
│   │   └── input_box.py            # Chat input with submit handling
│   └── styles/
│       └── app.tcss                # Textual CSS styles
│
├── agent/                          # ═══ THE AGENT (orchestration + LLM layer) ═══
│   ├── __init__.py
│   ├── loop.py                     # AgentLoop: while-loop + tool dispatch + streaming bridge
│   ├── orchestrator.py             # SessionOrchestrator: phase state machine, tool gating,
│   │                               #   forced tool calls, transition logic, validation
│   ├── phases.py                   # PhaseConfig dataclass + TEACHER_PHASES / INTERVIEW_PHASES
│   ├── prompts.py                  # Scoped system prompts per phase
│   └── tools/                      # Tool shims (thin wrappers → MLRefresherAPI)
│       ├── __init__.py             # ToolRegistry: registers all tools, schema lookup by name
│       ├── content.py              # search_content, get_lesson, get_question → cli.api
│       ├── code.py                 # run_python, get_code_example → cli.api
│       ├── assessment.py           # generate_quiz (LLM), evaluate_answer (LLM),
│       │                           #   get_interview_question → cli.api
│       ├── state.py                # get_progress, update_progress, get_review_schedule → cli.api
│       └── presentation.py         # render_diagram → cli.api
│
├── # ... existing dirs unchanged (pytorch_refresher/, interview_questions/, docs/, data/) ...
```

**Dependency direction**: `tui/` → `agent/` → `cli/`. The CLI has zero knowledge of the TUI or the agent. The agent imports `cli.api.MLRefresherAPI`. The TUI imports the agent.

### 11.1 Entry Points

```bash
# ── CLI (standalone, no LLM required) ──
uv run mlr content search "attention mechanisms"       # Search content
uv run mlr content lesson pytorch/06                   # Read a lesson
uv run mlr code run --code "import torch; print(1+1)"  # Run Python code
uv run mlr progress show                               # View progress
uv run mlr progress review                             # Spaced repetition schedule
uv run mlr index build                                 # Build/rebuild content index
uv run mlr diagram show transformer_full               # Render a diagram

# ── TUI (interactive, requires ANTHROPIC_API_KEY) ──
uv run python -m tui                                   # Launch the TUI

# ── Makefile shortcuts ──
make tui        # Launch the TUI
make index      # Build/rebuild the content index
make cli        # Alias for `uv run mlr`
```

```toml
# pyproject.toml — register the CLI as a script
[project.scripts]
mlr = "cli.__main__:mlr"
```

---

## 12. Implementation Phases

### Phase 1: CLI Foundation
**Goal**: Working `mlr` CLI with content retrieval and code execution. No LLM, no TUI.

- Click-based CLI with subcommand groups (`content`, `code`, `diagram`, `index`)
- `MLRefresherAPI` programmatic interface
- `mlr content lesson` — load and display lesson content
- `mlr content topics` / `mlr content questions` — list available content
- `mlr code run` — execute Python code via subprocess
- `mlr code example` — retrieve code examples from lesson files
- `mlr diagram show` / `mlr diagram list` — ASCII architecture diagrams
- `--json` flag on all commands
- `pyproject.toml` script entry point

**Ship criteria**: A human can use `mlr` to browse all lessons, run code examples, and view diagrams. All commands work with `--json` for machine consumption.

### Phase 2: RAG + Index
**Goal**: Semantic search over all content via the CLI.

- Content chunking pipeline (hierarchical: markdown, notebooks, Python files)
- LanceDB vector store + `nomic-embed-text-v1.5` embedding pipeline (256d Matryoshka)
- `mlr index build` / `mlr index status`
- `mlr content search` with topic-filtered retrieval
- Question bank extraction: parse interview questions into structured rubrics JSON
- `mlr content question` — retrieve question with rubric

**Ship criteria**: `mlr content search "attention mechanisms"` returns relevant, cited chunks from the book. `mlr content question attention_q2` returns the question, expected answer, and rubric.

### Phase 3: State + Progress
**Goal**: Persistent learning state via the CLI.

- SQLite schema (topics, learning_events, fsrs_cards)
- FSRS integration via `py-fsrs`
- `mlr progress show` / `mlr progress update` / `mlr progress review`
- Level tracking (novice → intermediate → advanced) based on score thresholds

**Ship criteria**: `mlr progress update --topic attention --event quiz --score 0.85` records the event, `mlr progress review` shows concepts due for review.

### Phase 4: Agent + TUI MVP
**Goal**: Working TUI with Teacher mode, powered by the CLI.

- Agent loop with Anthropic API streaming
- Tool shims that call `MLRefresherAPI` (not reimplementing CLI logic)
- Phase-based orchestrator (Teacher: WARMUP → INTRODUCE → EXPLORE → PRACTICE → WRAPUP)
- Tool gating and forced tool calls per phase
- Textual app with chat layout (sidebar + chat area + input)
- Streaming bridge: Anthropic API → Textual `MarkdownStream`
- Teacher system prompt with Socratic method instructions
- Topic navigation in sidebar

**Ship criteria**: User can select a topic, get a guided lesson with streaming responses, tool-grounded explanations (citing book content), and interactive quizzes.

### Phase 5: Interviewer + Assessment
**Goal**: Both personas functional with rubric-based evaluation.

- Interviewer phase state machine (SETUP → QUESTION → FOLLOWUP → EVALUATE → DEBRIEF)
- `evaluate_answer` agent tool (side-channel LLM call with rubric)
- `generate_quiz` agent tool (side-channel LLM call with content context)
- `mlr content next-question` — adaptive question selection via CLI
- Scoring display in UI
- Follow-up question logic in Interviewer prompt
- Adaptive difficulty (escalate/de-escalate based on scores)
- Mode switching in TUI (Teacher ↔ Interviewer)

**Ship criteria**: User can run a mock interview session, get scored against rubrics, and receive feedback with concept gap analysis.

### Phase 6: Polish
**Goal**: Production quality.

- Welcome screen with mode selection
- Progress dashboard screen (visual progress in TUI)
- Warm-up reviews at Teacher session start (spaced repetition)
- Cross-session knowledge state displayed in sidebar
- Error handling, rate limiting, graceful degradation on API failure
- Help text and onboarding flow
- CLI help text and man-page-style documentation

**Ship criteria**: Full feature set, polished UX, handles edge cases gracefully. Both CLI and TUI are documented.

---

## 13. Dependencies to Add

```toml
# pyproject.toml additions
[project]
dependencies = [
    # Existing deps...

    # CLI
    "click>=8.0",

    # TUI
    "textual>=4.0.0",

    # LLM
    "anthropic",

    # RAG — Embedding + Retrieval
    "lancedb",
    "sentence-transformers",
    "tantivy",                    # BM25 full-text search (used by LanceDB internally)

    # RAG — Re-ranking
    "onnxruntime",                # ONNX backend for cross-encoder CPU inference

    # Spaced Repetition
    "fsrs",
]

[project.scripts]
mlr = "cli.__main__:mlr"
```

---

## 14. Configuration

The TUI reads configuration from environment variables and/or a config file:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional
ML_TUTOR_MODEL=claude-sonnet-4-5-20250929    # Default model
ML_TUTOR_DB_PATH=~/.ml-refresher/state.db    # Learning state location
ML_TUTOR_INDEX_PATH=~/.ml-refresher/index/   # LanceDB storage
```

---

## 15. Open Questions

1. **Model selection**: Default to Sonnet for cost/speed balance, or let user choose? Teacher mode benefits from stronger reasoning (Opus), Interview mode is fine with Sonnet.
2. **Offline mode**: Should the TUI work without an API key (read-only content browsing, no agent)? Could be useful as a fallback.
3. **Export**: Should interview session transcripts be exportable (markdown file)? Useful for review.
4. **Multi-user**: Currently single-user (one SQLite DB). Worth supporting profiles for shared machines?
5. **Content hot-reload**: If the user adds new lessons/questions, should the index auto-update? Or require manual `make index`?

---

## 16. Success Metrics

- User can complete a full Teacher session (warm-up → lesson → quiz → wrap-up) in under 30 minutes
- Interview sessions score answers within 10% of what a human interviewer would rate (validated by manual review of 20 sessions)
- Spaced repetition increases quiz scores by 15%+ on review topics vs first-attempt scores
- Agent responses cite specific book content in >80% of explanatory answers
- Streaming response latency: first token appears in <1s after user input
