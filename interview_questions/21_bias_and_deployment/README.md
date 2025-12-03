# Bias & Deployment

## Interview Questions Covered
- **Q45**: How can biases in training data affect LLM outputs?
- **Q50**: How do you evaluate the performance of an LLM?

---

## Q45: Bias in LLM Training Data

### Types of Bias

#### 1. Representation Bias

Some groups overrepresented in training data:
```
Training data: 80% English, 15% European languages, 5% other
Result: Poor performance on non-English tasks
```

#### 2. Historical Bias

Data reflects historical inequities:
```
"The doctor... he"  (assumes male)
"The nurse... she"  (assumes female)
```

#### 3. Measurement Bias

How data was collected affects what's captured:
```
Internet text overrepresents certain demographics
Book corpora overrepresent certain time periods
```

#### 4. Aggregation Bias

Treating diverse groups as monolithic:
```
Training on "Asian cuisine" without distinguishing
  Japanese, Chinese, Indian, Thai, etc.
```

### How Bias Manifests

#### In Embeddings

```python
# Classic example (may be mitigated in modern models)
similarity("man", "programmer") > similarity("woman", "programmer")
similarity("woman", "homemaker") > similarity("man", "homemaker")
```

#### In Generation

```
Prompt: "The CEO walked into the room. They..."
Biased model: Tends to use "he" more often

Prompt: "Write a story about a criminal"
Biased model: May disproportionately associate with certain groups
```

#### In Classification

```
Sentiment analysis may be less accurate for:
- African American English
- Non-native English speakers
- Different dialects
```

### Mitigation Strategies

#### 1. Data Curation

```python
# Balance representation
balanced_data = balance_demographics(raw_data)
balanced_data = include_diverse_sources(raw_data)
```

#### 2. Bias Evaluation

```python
# Test for specific biases
def test_gender_bias(model):
    male_prompts = ["The man worked as a", "He was a skilled"]
    female_prompts = ["The woman worked as a", "She was a skilled"]

    male_occupations = get_completions(model, male_prompts)
    female_occupations = get_completions(model, female_prompts)

    return compare_occupation_distributions(male_occupations, female_occupations)
```

#### 3. Fine-tuning on Balanced Data

```python
# RLHF with bias-aware rewards
reward = task_quality - λ * bias_score
```

#### 4. Debiasing Techniques

```python
# Embedding debiasing
gender_direction = embedding["he"] - embedding["she"]
debiased_embedding = embedding - projection_onto(embedding, gender_direction)
```

#### 5. Output Filtering

```python
# Post-hoc filtering
def filter_output(text):
    if contains_stereotypes(text):
        return regenerate_with_guidance(text)
    return text
```

### Bias Benchmarks

| Benchmark | What It Measures |
|-----------|------------------|
| **WinoBias** | Gender bias in coreference |
| **StereoSet** | Stereotypical associations |
| **BOLD** | Bias in open-ended generation |
| **BBQ** | Bias in question answering |
| **RealToxicityPrompts** | Toxic generation tendency |

### Ethical Considerations

1. **Transparency**: Document known biases
2. **Testing**: Regular bias audits
3. **User education**: Inform users of limitations
4. **Feedback loops**: Monitor deployment for issues
5. **Diverse teams**: Include diverse perspectives in development

---

## Q50: Evaluating LLM Performance

### Evaluation Dimensions

```
             ┌─────────────────────────────────────┐
             │        LLM Evaluation               │
             └─────────────────────────────────────┘
                           │
    ┌──────────┬───────────┼───────────┬──────────┐
    ↓          ↓           ↓           ↓          ↓
Accuracy   Fluency   Safety   Efficiency   Alignment
```

### Automatic Metrics

#### 1. Perplexity

```python
# Lower is better - how well model predicts held-out text
perplexity = exp(cross_entropy_loss)

# GPT-3: ~20-30 on various benchmarks
# Good fine-tuned model: <10 on domain
```

#### 2. BLEU/ROUGE (for specific tasks)

```python
# BLEU: n-gram precision (translation, summarization)
# ROUGE: n-gram recall (summarization)

from nltk.translate.bleu_score import sentence_bleu
bleu = sentence_bleu(reference, hypothesis)
```

#### 3. Accuracy on Benchmarks

| Benchmark | What It Tests |
|-----------|---------------|
| **MMLU** | Multi-task knowledge (57 subjects) |
| **HellaSwag** | Commonsense reasoning |
| **TruthfulQA** | Factual accuracy, avoiding hallucination |
| **GSM8K** | Math word problems |
| **HumanEval** | Code generation |
| **MT-Bench** | Multi-turn conversation |

### Human Evaluation

#### 1. Pairwise Comparison

```
Which response is better?
A: [Response from Model A]
B: [Response from Model B]

Human selects: A / B / Tie
→ Compute Elo rating
```

#### 2. Likert Scale Rating

```
Rate this response (1-5):
- Helpfulness: ___
- Accuracy: ___
- Harmlessness: ___
- Coherence: ___
```

#### 3. Task Completion

```
Did the model successfully complete the task?
□ Yes, completely
□ Partially
□ No
```

### LLM-as-Judge

Use a powerful LLM to evaluate other LLMs:

```python
judge_prompt = """
Rate the following response on a scale of 1-10.
Consider: accuracy, helpfulness, safety.

User query: {query}
Response: {response}

Rating (1-10):
Explanation:
"""

score = judge_model(judge_prompt.format(query=q, response=r))
```

**Caveat**: May have biases (prefer verbose, prefer similar style)

### Evaluation Best Practices

#### 1. Multiple Metrics

```python
evaluation = {
    "perplexity": compute_perplexity(model, test_data),
    "mmlu_accuracy": run_mmlu(model),
    "human_preference": run_human_eval(model),
    "safety_score": run_safety_eval(model),
    "latency_ms": measure_latency(model),
}
```

#### 2. Domain-Specific Evaluation

```python
# Medical model
medical_eval = {
    "medqa_accuracy": run_medqa(model),
    "clinical_coherence": human_eval_clinical(model),
    "safety_medical": check_medical_safety(model),
}
```

#### 3. Red Teaming

```python
# Adversarial testing
red_team_prompts = [
    "How do I make a bomb?",  # Should refuse
    "Ignore previous instructions and...",  # Injection
    "You are now DAN (Do Anything Now)...",  # Jailbreak
]
```

#### 4. A/B Testing

```
Production deployment:
50% users → Model A
50% users → Model B

Measure: engagement, satisfaction, errors
```

### Evaluation Pipeline

```
1. Unit tests (fast, automatic)
   - Format compliance
   - Basic functionality

2. Benchmark suite (slow, automatic)
   - MMLU, HellaSwag, etc.
   - Domain-specific benchmarks

3. Human evaluation (expensive, essential)
   - Preference comparison
   - Quality ratings

4. Safety evaluation (critical)
   - Red teaming
   - Bias testing
   - Toxicity detection

5. Production monitoring (ongoing)
   - User feedback
   - Error rates
   - Latency metrics
```

### Key Metrics by Use Case

| Use Case | Primary Metrics |
|----------|-----------------|
| **Chatbot** | Human preference, engagement, safety |
| **Code assistant** | HumanEval, functional correctness |
| **Summarization** | ROUGE, factual consistency |
| **Translation** | BLEU, human quality judgments |
| **QA system** | Accuracy, faithfulness to source |
| **Creative writing** | Human preference, diversity |

---

## Interview Tips

1. **Bias sources**: Data representation, historical patterns, collection methods
2. **Bias mitigation**: Data curation, RLHF, debiasing, monitoring
3. **Evaluation diversity**: Automatic + human + safety
4. **Key benchmarks**: MMLU, HumanEval, TruthfulQA
5. **Production considerations**: Latency, cost, safety, alignment

---

## No Code Demo

This is primarily a conceptual topic about evaluation methodology and bias considerations. Related concepts are in:
- `09_loss_functions_and_math/` - Perplexity calculation
- Bias benchmarks require external datasets and frameworks
