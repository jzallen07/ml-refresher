# Model Quick Start Guide

Evaluated 6 models via OpenRouter for the ML Refresher TUI's learning and interview modes. Each model was tested with a 2-turn PyTorch gradients lesson and a 2-turn attention mechanism interview session.

## Ratings

| Model | Learning (1-5) | Interview (1-5) | Notes |
|-------|:--------------:|:---------------:|-------|
| `anthropic/claude-opus-4.6` | 5 | 5 | Best overall — clear explanations with visual graphs, intuitive analogies, thorough interview feedback |
| `anthropic/claude-sonnet-4.6` | 5 | 5 | Nearly as strong as Opus — slightly more concise, gives numeric scores (6/10) in interview mode |
| `minimax/minimax-m2.5` | 5 | 4 | Excellent teaching with tables and diagrams, solid interview feedback with actionable suggestions |
| `z-ai/glm-5` | 4 | 4 | Good explanations with visual breakdowns, interview feedback was cut short by token limit |
| `moonshotai/kimi-k2-0905` | 4 | 3 | Concise and accurate learning responses, interview coaching was terse — pushed for retry rather than giving feedback |
| `moonshotai/kimi-k2.5` | 2 | 4 | Empty response on learning turn 2 (thinking model token issue), interview coaching was detailed |

## Strengths & Weaknesses

**anthropic/claude-opus-4.6** — Best all-around. Computational graph ASCII art, chain rule walkthrough, structured interview feedback with clear categories. Slightly verbose.

**anthropic/claude-sonnet-4.6** — Close second. Faster and cheaper than Opus with comparable quality. Gives structured rubric-style feedback with scores. Good balance of depth and conciseness.

**minimax/minimax-m2.5** — Strong teaching model. Uses comparison tables, code examples with/without `requires_grad`, and practical "when to use" guidance. Interview feedback includes specific missing elements.

**z-ai/glm-5** — Solid fundamentals. Clean visual breakdowns and clear step-by-step math. Responses may hit token limits on longer evaluations.

**moonshotai/kimi-k2-0905** — Accurate but minimalist. Good for quick answers. Interview mode challenged the student to improve rather than evaluating, which may or may not suit the TUI's needs.

**moonshotai/kimi-k2.5** — Unreliable for learning mode (empty response). Likely a thinking/reasoning model where internal chain-of-thought consumes the token budget. Interview coaching was actually strong when it produced output.

## Recommended Default

**`anthropic/claude-sonnet-4.6`** — Best balance of quality, speed, and cost. Nearly matches Opus quality at lower latency and price. Use Opus for complex multi-turn sessions where depth matters most.

For budget-conscious usage, **`minimax/minimax-m2.5`** is a strong alternative with good teaching and interview capabilities.
