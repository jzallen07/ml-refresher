# Prompt Engineering

## Interview Questions Covered
- **Q13**: Why is prompt engineering crucial for LLM performance?
- **Q38**: What is Chain-of-Thought (CoT) prompting, and how does it aid reasoning?

---

## Q13: Why is Prompt Engineering Crucial?

### Definition

**Prompt engineering** is the practice of designing inputs to elicit desired behaviors from LLMs without modifying model weights.

### Why It Matters

1. **No training required**: Instant adaptation to new tasks
2. **Dramatic performance swings**: Same model, different prompt = vastly different results
3. **Cost effective**: Cheaper than fine-tuning
4. **Interpretable**: Can understand and iterate on prompts

### Key Techniques

#### 1. Clear Instructions
```
❌ "Summarize this"
✅ "Summarize this article in 3 bullet points, each under 20 words"
```

#### 2. Role/Persona Setting
```
"You are an expert Python developer. Review this code for bugs and security issues."
```

#### 3. Output Format Specification
```
"Respond in JSON format with keys: 'sentiment', 'confidence', 'explanation'"
```

#### 4. Few-Shot Examples
```
Classify the sentiment:
"Great product!" → Positive
"Terrible experience" → Negative
"The package arrived" → Neutral
"Best purchase ever!" → ?
```

#### 5. Constraints and Guardrails
```
"Answer only based on the provided context. If the answer isn't in the context, say 'I don't know'."
```

### Zero-Shot vs Few-Shot

| Approach | Description | When to Use |
|----------|-------------|-------------|
| **Zero-shot** | No examples, just instructions | Simple tasks |
| **One-shot** | One example | Need format clarity |
| **Few-shot** | 3-10 examples | Complex or ambiguous tasks |

### Common Mistakes

1. **Vague instructions**: "Do better" vs "Fix the grammar errors"
2. **Overloading**: Too many requirements in one prompt
3. **Assuming context**: LLMs don't remember previous conversations
4. **Ignoring output format**: Not specifying structure

---

## Q38: Chain-of-Thought (CoT) Prompting

### Definition

**Chain-of-Thought** prompting instructs the model to solve problems step-by-step, showing its reasoning process before giving the final answer.

### Standard vs CoT

**Standard Prompting**:
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many does he have?
A: 11
```

**Chain-of-Thought**:
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many does he have?
A: Let's think step by step.
   1. Roger starts with 5 balls
   2. He buys 2 cans with 3 balls each
   3. 2 cans × 3 balls = 6 new balls
   4. Total = 5 + 6 = 11 balls
   The answer is 11.
```

### Why CoT Works

1. **Decomposition**: Complex problems broken into simpler steps
2. **Working memory**: Intermediate results stored in context
3. **Error checking**: Mistakes visible in reasoning chain
4. **Emergent ability**: Only works well in large models (>100B params)

### CoT Variants

#### Zero-Shot CoT
Just add "Let's think step by step":
```
Q: [problem]
A: Let's think step by step.
```

#### Few-Shot CoT
Provide examples with reasoning:
```
Q: [example problem 1]
A: [step-by-step solution 1]

Q: [example problem 2]
A: [step-by-step solution 2]

Q: [actual problem]
A:
```

#### Self-Consistency CoT
1. Generate multiple reasoning chains
2. Take majority vote on final answer
3. More robust than single chain

### When to Use CoT

| Task Type | CoT Helpful? |
|-----------|--------------|
| Math word problems | Very helpful |
| Multi-step reasoning | Very helpful |
| Code debugging | Helpful |
| Simple factual questions | Not needed |
| Creative writing | Not needed |

### Limitations

1. **Longer outputs**: More tokens = higher cost
2. **Can confabulate**: Confident but wrong reasoning
3. **Model size**: Doesn't work well on small models
4. **Not always better**: Simple tasks may get worse

---

## Advanced Prompting Techniques

### Tree of Thoughts (ToT)
Explore multiple reasoning paths, backtrack if needed.

### ReAct (Reasoning + Acting)
Interleave reasoning with tool use:
```
Thought: I need to search for X
Action: Search[X]
Observation: [result]
Thought: Now I know Y, I should...
```

### Self-Ask
Model asks and answers its own sub-questions:
```
Q: Who was president when the iPhone launched?
Follow-up: When did the iPhone launch?
Answer: 2007
Follow-up: Who was US president in 2007?
Answer: George W. Bush
Final answer: George W. Bush
```

---

## Code Demo

See `cot_prompting_example.py` for:
- Standard vs CoT prompt comparison
- Few-shot CoT template builder
- Self-consistency implementation
- Prompt template library

```bash
python interview_questions/13_prompt_engineering/cot_prompting_example.py
```
