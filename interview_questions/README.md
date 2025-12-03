# LLM Interview Questions

A comprehensive study guide covering 50 essential Large Language Model interview questions, organized by topic. Each category includes detailed explanations and, where applicable, runnable code demonstrations.

> **Source**: Based on "Top 50 Large Language Model (LLM) Interview Questions" by Hao Hoang (May 2025).

## Categories

| # | Category | Questions | Code Demo |
|---|----------|-----------|-----------|
| 01 | [Tokenization & Text Processing](./01_tokenization_and_text/) | Q1, Q16 | Yes |
| 02 | [Attention Mechanisms](./02_attention_mechanisms/) | Q2, Q22-24, Q32 | Yes |
| 03 | [Transformer Architecture](./03_transformer_architecture/) | Q17, Q21, Q43, Q46 | Yes |
| 04 | [Context & Memory](./04_context_and_memory/) | Q3 | No |
| 05 | [Text Generation Strategies](./05_text_generation/) | Q5, Q6, Q12 | Yes |
| 06 | [Training Objectives](./06_training_objectives/) | Q7, Q9, Q11 | Yes |
| 07 | [Embeddings](./07_embeddings/) | Q10 | Yes |
| 08 | [Seq2Seq Models](./08_seq2seq_models/) | Q8 | No |
| 09 | [Loss Functions & Math](./09_loss_functions_and_math/) | Q25, Q29-31 | Yes |
| 10 | [Gradients & Optimization](./10_gradients_and_optimization/) | Q26, Q27, Q48 | Yes |
| 11 | [Fine-Tuning Methods](./11_fine_tuning_methods/) | Q4, Q14, Q35 | Yes |
| 12 | [Model Distillation](./12_model_distillation/) | Q15 | No |
| 13 | [Prompt Engineering](./13_prompt_engineering/) | Q13, Q38 | Yes |
| 14 | [Zero & Few-Shot Learning](./14_zero_and_few_shot/) | Q41, Q44 | No |
| 15 | [Regularization](./15_regularization/) | Q18 | Yes |
| 16 | [Generative vs Discriminative](./16_generative_vs_discriminative/) | Q19, Q39 | No |
| 17 | [Model Architectures](./17_model_architectures/) | Q20, Q33, Q34, Q37, Q47, Q49 | No |
| 18 | [RAG & Knowledge Graphs](./18_rag_and_knowledge/) | Q36, Q40 | Yes |
| 19 | [Efficiency & Scaling](./19_efficiency_and_scaling/) | Q42 | No |
| 20 | [Dimensionality Reduction](./20_dimensionality_reduction/) | Q28 | Yes |
| 21 | [Bias & Deployment](./21_bias_and_deployment/) | Q45, Q50 | No |

## How to Use This Guide

1. **Study by category**: Each directory contains a README with comprehensive explanations
2. **Run the demos**: Categories with code have runnable `.py` files demonstrating concepts
3. **Practice explaining**: Try explaining each concept out loud as interview practice

## Running Code Demos

```bash
# Activate poetry environment
poetry shell

# Run any demo
python interview_questions/01_tokenization_and_text/tokenization_demo.py
python interview_questions/02_attention_mechanisms/attention_demo.py
# etc.
```

## Quick Reference: All 50 Questions

### Fundamentals
- Q1: What is tokenization and why is it critical?
- Q10: What are embeddings and how are they initialized?
- Q49: What defines a Large Language Model?

### Architecture
- Q2: How does attention work in transformers?
- Q17: How do transformers improve on Seq2Seq?
- Q21: What are positional encodings?
- Q22: What is multi-head attention?
- Q46: How do encoders and decoders differ?

### Training
- Q7: What is masked language modeling?
- Q9: Autoregressive vs masked models?
- Q11: What is next sentence prediction?
- Q25: Why cross-entropy loss for language modeling?

### Generation
- Q5: Beam search vs greedy decoding?
- Q6: What role does temperature play?
- Q12: Top-k vs top-p sampling?

### Fine-Tuning & Adaptation
- Q4: LoRA vs QLoRA?
- Q14: How to avoid catastrophic forgetting?
- Q35: How does PEFT work?
- Q41: What is zero-shot learning?
- Q44: What is few-shot learning?

### Advanced Topics
- Q36: What is RAG?
- Q37: How does Mixture of Experts work?
- Q38: What is Chain-of-Thought prompting?
- Q40: How do knowledge graphs improve LLMs?

### Math & Optimization
- Q23: Softmax in attention?
- Q24: Dot product in self-attention?
- Q29: What is KL divergence?
- Q30: ReLU derivative?
- Q31: Chain rule in gradient descent?
