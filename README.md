# ML Refresher

A comprehensive learning repository for Machine Learning concepts, PyTorch fundamentals, and LLM interview preparation — with an interactive TUI powered by Claude.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- `OPENROUTER_API_KEY` environment variable (for interactive sessions)

## Setup

```bash
# Install dependencies
uv sync

# Set your API key
export OPENROUTER_API_KEY=sk-or-...

# Build the search index (first time only)
uv run mlr index build
```

## Interactive Sessions

The `mlr` CLI provides two LLM-powered session modes that use a phased agentic architecture with Claude:

### Teacher Mode

Socratic tutor that guides you through lessons, adapts to your level, and runs live code demos.

```bash
uv run mlr learn attention_mechanisms
```

Phases: warmup → introduce → explore → practice → wrapup

### Interviewer Mode

Mock ML interviewer that asks adaptive questions from the 50+ question bank, evaluates your answers with rubric scoring, and gives detailed feedback.

```bash
uv run mlr interview attention_mechanisms
```

Phases: setup → question → followup → evaluate → debrief (loops for 3 questions)

### TUI

Launch the full terminal UI with mode/topic selection, a live sidebar showing phase progress, and a progress dashboard (`ctrl+p`):

```bash
uv run mlr tui
```

Both modes share persistent state across sessions — FSRS spaced repetition scheduling tracks concept mastery and prioritizes weak areas for review.

## CLI Reference

```bash
uv run mlr content topics                          # List all topics
uv run mlr content questions attention_mechanisms   # List questions in a topic
uv run mlr content next-question attention_mechanisms  # FSRS-aware question selection
uv run mlr content search "transformer architecture"   # Semantic + keyword search
uv run mlr progress show                            # View learning progress
uv run mlr progress schedule                        # View spaced-repetition schedule
uv run mlr code run <file>                          # Execute a code example
uv run mlr diagram list                             # List available diagrams
uv run mlr index build                              # Build/rebuild search index
```

## Interactive Book UI

This repo also includes Jupyter Book for a browser-based reading experience:

```bash
# Start the local book server (read-only)
uv run jupyter-book start
```

### With Live Code Execution

To run code interactively in the browser, start a Jupyter server alongside the book:

```bash
# Terminal 1: Start Jupyter server for code execution
uv run jupyter lab --NotebookApp.token=mlrefresher --NotebookApp.allow_origin='*'

# Terminal 2: Start the book UI
uv run jupyter-book start
```

Then click the "power" button on any notebook page to connect to the Jupyter kernel and run code cells.

The book auto-discovers new content - just add a folder with a `README.md` and it appears in the navigation.

## Repository Structure

```
ml-refresher/
  agent/                 # Agentic LLM layer
    tools/               #   Tool implementations (12 tools)
    orchestrator.py      #   Phase-based session orchestrator
    loop.py              #   Claude API agent loop with streaming
    phases.py            #   Teacher & interviewer phase configs
    prompts.py           #   System prompts per phase
    harness.py           #   Tool registry builder
  tui/                   # Textual TUI application
    app.py               #   Main app with chat interface
    screens.py           #   Welcome screen, progress dashboard
    widgets.py           #   Sidebar, tool indicators
    bridge.py            #   TUI ↔ agent integration
  cli/                   # CLI entry point (mlr command)
    commands/            #   Click command groups
    services/            #   Content parsing, search, code execution
    state/               #   SQLite state, FSRS scheduling, progress
    rubrics/             #   Question rubrics with scoring criteria
    api.py               #   Unified API surface for all operations
  pytorch_refresher/     # PyTorch fundamentals (10 lessons)
  interview_questions/   # LLM interview prep (50 questions, 21 categories)
  data/                  # Generated outputs and visualizations
  docs/                  # TUI PRD and model guide
  scripts/               # Evaluation and utility scripts
```

## PyTorch Refresher

Progressive lessons covering PyTorch fundamentals:

| Lesson | Topic | Description |
|--------|-------|-------------|
| [01](./pytorch_refresher/01_tensors/) | Tensors | Creation, operations, memory layout |
| [02](./pytorch_refresher/02_autograd/) | Autograd | Automatic differentiation, gradients |
| [03](./pytorch_refresher/03_neural_networks/) | Neural Networks | nn.Module, layers, forward pass |
| [04](./pytorch_refresher/04_training_loop/) | Training Loop | Loss, optimizers, batch training |
| [05](./pytorch_refresher/05_activation_functions/) | Activation Functions | ReLU, Sigmoid, GELU, comparisons |
| [06](./pytorch_refresher/06_loss_functions/) | Loss Functions | MSE, CrossEntropy, custom losses |
| [07](./pytorch_refresher/07_data_management/) | Data Management | Datasets, transforms, augmentation |
| [08](./pytorch_refresher/08_saving_loading/) | Saving & Loading | Checkpoints, state dicts |
| [09](./pytorch_refresher/09_dataloaders/) | DataLoaders | Batching, shuffling, workers |
| [10](./pytorch_refresher/10_gpu_training/) | GPU Training | CUDA, device management |

```bash
# Run a lesson
uv run python pytorch_refresher/01_tensors/lesson.py
```

## LLM Interview Questions

50 interview questions organized into 21 thematic categories with detailed explanations and code demos.

| Category | Questions | Code Demo |
|----------|-----------|-----------|
| [Tokenization & Text](./interview_questions/01_tokenization_and_text/) | Q1, Q16 | Yes |
| [Attention Mechanisms](./interview_questions/02_attention_mechanisms/) | Q2, Q22-24, Q32 | Yes |
| [Transformer Architecture](./interview_questions/03_transformer_architecture/) | Q17, Q21, Q43, Q46 | Yes |
| [Context & Memory](./interview_questions/04_context_and_memory/) | Q3 | - |
| [Text Generation](./interview_questions/05_text_generation/) | Q5, Q6, Q12 | Yes |
| [Training Objectives](./interview_questions/06_training_objectives/) | Q7, Q9, Q11 | Yes |
| [Embeddings](./interview_questions/07_embeddings/) | Q10 | Yes |
| [Seq2Seq Models](./interview_questions/08_seq2seq_models/) | Q8 | - |
| [Loss Functions & Math](./interview_questions/09_loss_functions_and_math/) | Q25, Q29-31 | Yes |
| [Gradients & Optimization](./interview_questions/10_gradients_and_optimization/) | Q26, Q27, Q48 | Yes |
| [Fine-Tuning (LoRA/PEFT)](./interview_questions/11_fine_tuning_methods/) | Q4, Q14, Q35 | Yes |
| [Model Distillation](./interview_questions/12_model_distillation/) | Q15 | - |
| [Prompt Engineering](./interview_questions/13_prompt_engineering/) | Q13, Q38 | Yes |
| [Zero & Few-Shot](./interview_questions/14_zero_and_few_shot/) | Q41, Q44 | - |
| [Regularization](./interview_questions/15_regularization/) | Q18 | Yes |
| [Generative vs Discriminative](./interview_questions/16_generative_vs_discriminative/) | Q19, Q39 | - |
| [Model Architectures](./interview_questions/17_model_architectures/) | Q20, Q33, Q34, Q37, Q47, Q49 | - |
| [RAG & Knowledge](./interview_questions/18_rag_and_knowledge/) | Q36, Q40 | Yes |
| [Efficiency & Scaling](./interview_questions/19_efficiency_and_scaling/) | Q42 | - |
| [Dimensionality Reduction](./interview_questions/20_dimensionality_reduction/) | Q28 | Yes |
| [Bias & Deployment](./interview_questions/21_bias_and_deployment/) | Q45, Q50 | - |

```bash
# Run an interview demo
uv run python interview_questions/01_tokenization_and_text/tokenization_demo.py
```

See [interview_questions/README.md](./interview_questions/README.md) for the complete guide.

## Generated Visualizations

Code demos generate educational visualizations saved to `data/interview_viz/`:

- Attention patterns and heatmaps
- Loss landscapes and training curves
- Embedding projections (PCA, t-SNE)
- Dropout and regularization effects
- RAG pipeline diagrams

## Source Materials

- PyTorch lessons based on TK's "Mastering PyTorch" article
- Interview questions from "Top 50 LLM Interview Questions" by Hao Hoang
