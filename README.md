# ML Refresher

A comprehensive learning repository for Machine Learning concepts, PyTorch fundamentals, and LLM interview preparation.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

## Setup

```bash
# Install dependencies
uv sync

# Run any script
uv run python <script_path>
```

## Repository Structure

```
ml-refresher/
  pytorch_refresher/     # PyTorch fundamentals (10 lessons)
  interview_questions/   # LLM interview prep (50 questions, 21 categories)
  data/                  # Generated outputs and visualizations
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
