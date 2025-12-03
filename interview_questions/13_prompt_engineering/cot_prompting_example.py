"""
Chain-of-Thought (CoT) Prompting for LLM Interviews

This module demonstrates various prompt engineering techniques, particularly
Chain-of-Thought prompting, which is crucial for improving LLM reasoning capabilities.

Key Interview Topics (Q13, Q38):
================================
1. Standard vs Chain-of-Thought Prompting
2. Zero-shot CoT ("Let's think step by step")
3. Few-shot CoT with examples
4. Self-consistency sampling
5. Prompt template best practices
6. Task-specific prompt patterns

Core Concepts:
=============
- Chain-of-Thought: Prompting LLMs to show intermediate reasoning steps
- Few-shot Learning: Providing examples in the prompt
- Zero-shot: No examples, just instructions
- Self-consistency: Multiple reasoning paths → majority vote
- Prompt Engineering: Crafting effective inputs to guide LLM behavior

Why CoT Works:
=============
1. Breaks down complex reasoning into manageable steps
2. Makes the model's reasoning process explicit and verifiable
3. Reduces errors by preventing "jumping to conclusions"
4. Enables better performance on arithmetic, logic, and multi-step problems
5. Allows humans to audit the reasoning process

Research Background:
===================
- Wei et al. (2022): "Chain-of-Thought Prompting Elicits Reasoning in LLMs"
- Kojima et al. (2022): "Large Language Models are Zero-Shot Reasoners"
- Wang et al. (2022): "Self-Consistency Improves Chain of Thought Reasoning"

Interview Talking Points:
========================
1. CoT improves performance on complex reasoning tasks (math, logic, common sense)
2. Zero-shot CoT works surprisingly well with just "Let's think step by step"
3. Few-shot CoT benefits from diverse, high-quality examples
4. Self-consistency can improve accuracy by 10-20% over greedy decoding
5. CoT adds latency/cost due to longer outputs (important production trade-off)
6. Not all tasks benefit equally - simple tasks may not need CoT
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# SECTION 1: Standard vs Chain-of-Thought Prompts
# ============================================================================

class PromptStyle(Enum):
    """Different prompting styles for comparison."""
    STANDARD = "standard"
    ZERO_SHOT_COT = "zero_shot_cot"
    FEW_SHOT_COT = "few_shot_cot"
    SELF_CONSISTENCY = "self_consistency"


def standard_prompt_example():
    """
    Standard prompting: Direct question → direct answer.

    Limitations:
    - No intermediate reasoning shown
    - Model may make logical leaps
    - Hard to debug when wrong
    - Lower accuracy on complex problems
    """
    prompt = """
Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?

Answer:
""".strip()

    expected_output = "11 tennis balls"

    return prompt, expected_output


def chain_of_thought_prompt_example():
    """
    Chain-of-Thought prompting: Show reasoning steps before answer.

    Advantages:
    - Explicit intermediate reasoning
    - Easier to verify correctness
    - Better performance on multi-step problems
    - More interpretable results

    Key Insight: The model learns to break down problems by seeing examples
    that demonstrate step-by-step reasoning.
    """
    prompt = """
Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?

Answer: Let me work through this step-by-step:
1. Roger starts with 5 tennis balls
2. He buys 2 cans of tennis balls
3. Each can contains 3 tennis balls
4. So he buys 2 × 3 = 6 tennis balls
5. Total tennis balls = 5 + 6 = 11

Therefore, Roger has 11 tennis balls now.
""".strip()

    return prompt


# ============================================================================
# SECTION 2: Zero-Shot Chain-of-Thought
# ============================================================================

def zero_shot_cot_template(question: str) -> str:
    """
    Zero-shot CoT: Add "Let's think step by step" to trigger reasoning.

    Discovery (Kojima et al., 2022):
    Simply appending "Let's think step by step" dramatically improves
    reasoning on many tasks WITHOUT any examples.

    Why it works:
    - The phrase triggers the model's learned reasoning patterns
    - Encourages decomposition of complex problems
    - No examples needed (reduces prompt length/cost)

    When to use:
    - Math word problems
    - Logic puzzles
    - Multi-hop reasoning
    - Common sense reasoning

    Interview Note: This is one of the most impactful prompt engineering
    discoveries - simple but highly effective.
    """
    return f"""
{question}

Let's think step by step:
""".strip()


def zero_shot_cot_examples():
    """Demonstrate zero-shot CoT on different problem types."""

    examples = {
        "arithmetic": {
            "question": "If a shirt costs $15 and is on sale for 20% off, what is the final price?",
            "prompt": zero_shot_cot_template(
                "If a shirt costs $15 and is on sale for 20% off, what is the final price?"
            ),
            "reasoning": """
1. Original price: $15
2. Discount: 20% of $15 = 0.20 × $15 = $3
3. Final price: $15 - $3 = $12

Answer: $12
"""
        },

        "logic": {
            "question": "All roses are flowers. Some flowers fade quickly. Does it follow that some roses fade quickly?",
            "prompt": zero_shot_cot_template(
                "All roses are flowers. Some flowers fade quickly. Does it follow that some roses fade quickly?"
            ),
            "reasoning": """
1. Given: All roses are flowers (roses ⊆ flowers)
2. Given: Some flowers fade quickly (∃ flowers that fade quickly)
3. Question: Do some roses fade quickly?
4. The second statement tells us some flowers fade quickly, but doesn't specify which flowers
5. Roses are flowers, but we don't know if the quickly-fading flowers include roses
6. This is invalid logical inference

Answer: No, it does not follow. The quickly-fading flowers might not include any roses.
"""
        },

        "common_sense": {
            "question": "John put his laptop in his backpack and went to the library. When he arrived, his laptop was gone. What likely happened?",
            "prompt": zero_shot_cot_template(
                "John put his laptop in his backpack and went to the library. When he arrived, his laptop was gone. What likely happened?"
            ),
            "reasoning": """
1. John put laptop in backpack (confirmed)
2. Laptop disappeared between home and library
3. Possible explanations:
   - Backpack zipper was open → laptop fell out
   - Laptop was stolen from backpack
   - John forgot to close backpack properly
   - Someone accessed his backpack during transit
4. Most common scenario: open/insecure backpack allowing loss/theft

Answer: The laptop likely fell out of an open or improperly closed backpack,
or was stolen if the backpack was left unattended.
"""
        }
    }

    return examples


# ============================================================================
# SECTION 3: Few-Shot Chain-of-Thought
# ============================================================================

@dataclass
class CoTExample:
    """Structure for a Chain-of-Thought example."""
    question: str
    reasoning: str
    answer: str


def build_few_shot_cot_prompt(
    examples: List[CoTExample],
    test_question: str,
    instruction: str = "Answer the following questions with step-by-step reasoning."
) -> str:
    """
    Build a few-shot CoT prompt with examples.

    Best Practices:
    ===============
    1. Use 3-8 examples (more examples = better but more expensive)
    2. Examples should be diverse (cover different problem types)
    3. Show consistent reasoning format across examples
    4. Include both simple and complex examples
    5. Make sure reasoning is correct and clear

    Structure:
    ==========
    [Instruction]
    [Example 1: Question → Reasoning → Answer]
    [Example 2: Question → Reasoning → Answer]
    ...
    [Test Question]

    Interview Note: Quality of examples matters more than quantity.
    Bad examples can hurt performance!
    """
    prompt_parts = [instruction, ""]

    # Add examples
    for i, example in enumerate(examples, 1):
        prompt_parts.append(f"Question {i}: {example.question}")
        prompt_parts.append(f"\nReasoning: {example.reasoning}")
        prompt_parts.append(f"\nAnswer: {example.answer}")
        prompt_parts.append("")  # Empty line between examples

    # Add test question
    prompt_parts.append(f"Question {len(examples) + 1}: {test_question}")
    prompt_parts.append("\nReasoning:")

    return "\n".join(prompt_parts)


def few_shot_cot_math_examples() -> List[CoTExample]:
    """
    High-quality few-shot examples for math word problems.

    Example Selection Strategy:
    - Mix of operations (addition, subtraction, multiplication, division)
    - Different complexity levels
    - Clear, consistent reasoning format
    - Explicitly show calculations
    """
    return [
        CoTExample(
            question="A cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?",
            reasoning="""
Step 1: Start with initial amount: 23 apples
Step 2: Subtract apples used: 23 - 20 = 3 apples remaining
Step 3: Add newly bought apples: 3 + 6 = 9 apples
""",
            answer="9 apples"
        ),

        CoTExample(
            question="Sarah has 3 boxes of cookies. Each box contains 12 cookies. She gives away 8 cookies. How many does she have left?",
            reasoning="""
Step 1: Calculate total cookies: 3 boxes × 12 cookies/box = 36 cookies
Step 2: Subtract cookies given away: 36 - 8 = 28 cookies
""",
            answer="28 cookies"
        ),

        CoTExample(
            question="A parking lot has 4 levels. Each level can hold 15 cars. If 47 cars are parked, how many spaces are free?",
            reasoning="""
Step 1: Calculate total capacity: 4 levels × 15 cars/level = 60 spaces
Step 2: Calculate free spaces: 60 - 47 = 13 spaces
""",
            answer="13 free spaces"
        )
    ]


def few_shot_cot_logic_examples() -> List[CoTExample]:
    """Few-shot examples for logical reasoning tasks."""
    return [
        CoTExample(
            question="If all mammals are warm-blooded and all dogs are mammals, are all dogs warm-blooded?",
            reasoning="""
Step 1: Identify premises:
  - P1: All mammals are warm-blooded (mammals → warm-blooded)
  - P2: All dogs are mammals (dogs → mammals)
Step 2: Apply transitive property:
  - If dogs → mammals AND mammals → warm-blooded
  - Then dogs → warm-blooded
Step 3: This is valid deductive reasoning (modus ponens)
""",
            answer="Yes, all dogs are warm-blooded."
        ),

        CoTExample(
            question="Some birds can fly. Penguins are birds. Can penguins fly?",
            reasoning="""
Step 1: Identify premises:
  - P1: Some birds can fly (∃ birds that fly)
  - P2: Penguins are birds (penguins ⊆ birds)
Step 2: Analyze logical relationship:
  - "Some birds" does not mean "all birds"
  - We cannot conclude properties of specific birds from "some birds"
Step 3: This is invalid reasoning (existential quantifier doesn't apply to all)
""",
            answer="No, we cannot conclude that penguins can fly. In fact, penguins cannot fly despite being birds."
        )
    ]


# ============================================================================
# SECTION 4: Self-Consistency with CoT
# ============================================================================

def self_consistency_template(question: str, num_paths: int = 5) -> str:
    """
    Self-consistency: Sample multiple reasoning paths and take majority vote.

    Algorithm (Wang et al., 2022):
    ==============================
    1. Generate multiple CoT reasoning paths (using temperature > 0)
    2. Extract final answer from each path
    3. Take majority vote among answers
    4. Return most common answer

    Why it works:
    =============
    - Different reasoning paths may catch different errors
    - Correct answers tend to be more consistent across paths
    - Reduces impact of single-path errors
    - Works best with diverse reasoning (higher temperature)

    Performance Gains:
    ==================
    - Typically 5-15% accuracy improvement over single CoT
    - Diminishing returns after 10-20 samples
    - Trade-off: 5-10x more API calls/cost

    Production Considerations:
    =========================
    - Use for high-stakes decisions where accuracy > cost
    - Can parallelize API calls for lower latency
    - Consider caching if same questions asked repeatedly

    Interview Note: This is a classic bias-variance trade-off.
    More samples reduce variance (improve robustness) at higher cost.
    """
    return f"""
I will solve this problem multiple times using different reasoning approaches,
then determine the most consistent answer.

Problem: {question}

Please provide {num_paths} different reasoning paths to solve this problem.
Each path should:
1. Show clear step-by-step reasoning
2. Arrive at a final answer
3. Use a potentially different approach or perspective

After all paths, identify the most common answer.
""".strip()


def self_consistency_explanation():
    """
    Detailed explanation of self-consistency for interviews.
    """
    explanation = """
=============================================================================
SELF-CONSISTENCY WITH CHAIN-OF-THOUGHT
=============================================================================

CONCEPT:
--------
Instead of using a single reasoning path (greedy decoding), generate multiple
diverse reasoning paths and aggregate their answers via majority voting.

ALGORITHM:
----------
```
Input: Question Q, LLM M, Number of samples N
Output: Final answer A

1. For i = 1 to N:
     Generate reasoning path R_i with high temperature (e.g., 0.7)
     Extract answer A_i from R_i

2. Count frequency of each unique answer
3. Return most frequent answer (majority vote)
```

EXAMPLE:
--------
Question: "If 3 cats catch 3 mice in 3 minutes, how many cats are needed
           to catch 100 mice in 100 minutes?"

Path 1 (Correct):
- Rate: 1 cat catches 1 mouse in 3 minutes
- In 100 minutes: 1 cat catches 100/3 ≈ 33 mice
- Need 3 cats to catch 100 mice in 100 minutes
Answer: 3 cats

Path 2 (Correct):
- 3 cats, 3 mice, 3 minutes → rate is constant
- Same setup, so same number of cats needed
Answer: 3 cats

Path 3 (Incorrect):
- More mice need more cats: 100/3 ≈ 33 cats
Answer: 33 cats

Path 4 (Correct):
- Each cat catches 1 mouse per 3 minutes
- In 100 minutes, each cat catches ~33 mice
- 3 cats catch ~100 mice
Answer: 3 cats

Path 5 (Correct):
- Ratio analysis: same time ratio = same cat ratio
Answer: 3 cats

MAJORITY VOTE: 3 cats (appears 4 times) → Final Answer

BENEFITS:
---------
✓ More robust to reasoning errors
✓ Builds confidence in answers (high agreement = high confidence)
✓ Can identify when model is uncertain (low agreement)
✓ Often improves accuracy by 10-20%

DRAWBACKS:
----------
✗ Higher computational cost (N times more expensive)
✗ Higher latency (unless parallelized)
✗ Not always necessary for simple questions
✗ Requires post-processing to extract and compare answers

WHEN TO USE:
------------
1. High-stakes decisions (medical, financial, legal)
2. Complex reasoning tasks (math, logic, planning)
3. When model confidence is important
4. When you can afford higher cost/latency

WHEN TO AVOID:
--------------
1. Simple questions with obvious answers
2. Real-time applications with strict latency requirements
3. Cost-sensitive applications
4. Tasks where reasoning diversity doesn't help

IMPLEMENTATION TIPS:
-------------------
1. Use temperature 0.7-0.8 for diversity (not too high or too low)
2. Start with N=5, increase if needed (diminishing returns after 10-20)
3. Parallelize API calls when possible
4. Implement robust answer extraction (regex, parsing, etc.)
5. Handle ties (could use confidence, answer length, etc.)

INTERVIEW INSIGHT:
-----------------
Self-consistency is an ensemble method applied to reasoning. Like random
forests in ML, it improves performance by aggregating diverse predictions.
The key innovation is recognizing that LLMs can generate diverse reasoning
paths, not just diverse final answers.
"""
    return explanation


# ============================================================================
# SECTION 5: Task-Specific Prompt Templates
# ============================================================================

class PromptTemplateLibrary:
    """
    Collection of proven prompt templates for common tasks.

    These templates encode best practices for different task types.
    Can be customized for specific use cases.
    """

    @staticmethod
    def classification_template(
        text: str,
        labels: List[str],
        description: str = "",
        few_shot_examples: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """
        Template for text classification tasks.

        Best Practices:
        - Clearly list all possible labels
        - Provide label descriptions if ambiguous
        - Use few-shot examples for better accuracy
        - Ask for reasoning before classification (improves accuracy)
        """
        prompt_parts = ["Text Classification Task"]

        if description:
            prompt_parts.append(f"\nTask Description: {description}")

        prompt_parts.append(f"\nPossible Labels: {', '.join(labels)}")

        # Add few-shot examples if provided
        if few_shot_examples:
            prompt_parts.append("\nExamples:")
            for example_text, example_label in few_shot_examples:
                prompt_parts.append(f"\nText: {example_text}")
                prompt_parts.append(f"Label: {example_label}")

        # Add test case
        prompt_parts.append(f"\nNow classify this text:")
        prompt_parts.append(f"\nText: {text}")
        prompt_parts.append("\nFirst, provide brief reasoning for your classification.")
        prompt_parts.append("Then provide the final label.")
        prompt_parts.append("\nReasoning:")

        return "\n".join(prompt_parts)

    @staticmethod
    def summarization_template(
        text: str,
        max_length: Optional[int] = None,
        style: str = "concise",
        focus: Optional[str] = None
    ) -> str:
        """
        Template for text summarization.

        Parameters:
        - max_length: Maximum words/sentences in summary
        - style: concise, detailed, bullet-points, etc.
        - focus: Specific aspect to focus on
        """
        prompt_parts = ["Please summarize the following text."]

        if max_length:
            prompt_parts.append(f"\nMaximum length: {max_length} words")

        if style:
            prompt_parts.append(f"Style: {style}")

        if focus:
            prompt_parts.append(f"Focus particularly on: {focus}")

        prompt_parts.append(f"\nText to summarize:\n{text}")
        prompt_parts.append("\nSummary:")

        return "\n".join(prompt_parts)

    @staticmethod
    def question_answering_template(
        context: str,
        question: str,
        answer_style: str = "concise"
    ) -> str:
        """
        Template for question answering with context.

        Best Practices:
        - Provide context before question
        - Specify desired answer format
        - Ask model to cite context (reduces hallucination)
        - Request "I don't know" if answer not in context
        """
        return f"""
Answer the question based on the context below.

Instructions:
- Only use information from the provided context
- If the answer is not in the context, say "I don't know"
- Provide a {answer_style} answer
- Cite specific parts of the context if possible

Context:
{context}

Question: {question}

Answer:
""".strip()

    @staticmethod
    def extraction_template(
        text: str,
        entities_to_extract: List[str],
        output_format: str = "json"
    ) -> str:
        """
        Template for information extraction tasks.

        Common use cases:
        - Named entity recognition
        - Structured data extraction
        - Key information retrieval
        """
        entities_str = ", ".join(entities_to_extract)

        return f"""
Extract the following information from the text below:
{entities_str}

Return the results in {output_format} format.
If any information is not found, use null or "Not found" as the value.

Text:
{text}

Extracted Information:
""".strip()

    @staticmethod
    def creative_generation_template(
        task: str,
        constraints: Optional[List[str]] = None,
        tone: str = "professional",
        examples: Optional[List[str]] = None
    ) -> str:
        """
        Template for creative text generation.

        Use cases:
        - Writing assistance
        - Content generation
        - Brainstorming
        """
        prompt_parts = [f"Task: {task}"]

        if tone:
            prompt_parts.append(f"\nTone: {tone}")

        if constraints:
            prompt_parts.append("\nConstraints:")
            for constraint in constraints:
                prompt_parts.append(f"- {constraint}")

        if examples:
            prompt_parts.append("\nStyle Examples:")
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"\nExample {i}: {example}")

        prompt_parts.append("\nGenerated Text:")

        return "\n".join(prompt_parts)

    @staticmethod
    def reasoning_template(
        problem: str,
        reasoning_type: str = "step-by-step"
    ) -> str:
        """
        Template for explicit reasoning tasks.

        Reasoning types:
        - step-by-step: Sequential reasoning
        - pros-cons: Decision making
        - cause-effect: Causal analysis
        - compare-contrast: Comparison
        """
        templates = {
            "step-by-step": f"""
Problem: {problem}

Please solve this problem using step-by-step reasoning:
1. Break down the problem into smaller parts
2. Solve each part sequentially
3. Show all intermediate steps
4. Arrive at a final answer

Solution:
""",
            "pros-cons": f"""
Question: {problem}

Analyze this by listing:
1. Pros (advantages, benefits, positive aspects)
2. Cons (disadvantages, drawbacks, negative aspects)
3. Final recommendation based on the analysis

Analysis:
""",
            "cause-effect": f"""
Situation: {problem}

Analyze the cause-and-effect relationships:
1. Identify the main causes
2. Explain the mechanisms
3. Describe the effects/consequences
4. Consider second-order effects

Analysis:
""",
            "compare-contrast": f"""
Items to compare: {problem}

Provide a detailed comparison:
1. Similarities between the items
2. Key differences
3. Relative advantages of each
4. Conclusion or recommendation

Comparison:
"""
        }

        return templates.get(reasoning_type, templates["step-by-step"]).strip()


# ============================================================================
# SECTION 6: Advanced Prompt Engineering Techniques
# ============================================================================

class AdvancedPromptTechniques:
    """
    Advanced techniques for prompt engineering.

    These go beyond basic CoT and are useful for specialized scenarios.
    """

    @staticmethod
    def least_to_most_prompting(problem: str, subproblems: List[str]) -> str:
        """
        Least-to-Most Prompting: Solve subproblems sequentially.

        Idea: Break complex problem into simpler subproblems, solve in order.
        Each subproblem's solution is used to solve the next.

        When to use:
        - Compositional problems
        - Problems with clear dependency structure
        - Mathematical proofs
        """
        prompt_parts = [
            "I'll solve this problem by breaking it into simpler subproblems.",
            f"\nMain Problem: {problem}",
            "\nSubproblems to solve in order:"
        ]

        for i, subproblem in enumerate(subproblems, 1):
            prompt_parts.append(f"{i}. {subproblem}")

        prompt_parts.append("\nLet's solve these one by one:")

        return "\n".join(prompt_parts)

    @staticmethod
    def program_aided_language_model(problem: str) -> str:
        """
        Program-Aided Language Model (PAL): Generate code to solve problem.

        Idea: LLM generates Python code, we execute code for final answer.
        Combines language understanding with precise computation.

        Benefits:
        - Perfect arithmetic (no calculation errors)
        - Complex computations possible
        - Verifiable logic

        When to use:
        - Math problems
        - Data analysis
        - Algorithmic problems
        """
        return f"""
Problem: {problem}

Generate Python code to solve this problem. The code should:
1. Define all necessary variables
2. Perform the required calculations
3. Print the final answer

Requirements:
- Use descriptive variable names
- Add comments explaining each step
- Ensure code is executable
- Handle edge cases

Python Code:
```python
""".strip()

    @staticmethod
    def react_prompting(task: str) -> str:
        """
        ReAct: Reasoning + Acting (for agents with tools).

        Idea: Interleave reasoning, action, and observation steps.

        Format:
        Thought: [reasoning about what to do]
        Action: [action to take, e.g., search Wikipedia]
        Observation: [result of action]
        ... (repeat)
        Thought: I now know the final answer
        Answer: [final answer]

        When to use:
        - Agent-based systems
        - Multi-step tasks requiring external info
        - Problems needing tool use
        """
        return f"""
Task: {task}

Solve this task using the Thought-Action-Observation framework:

Thought: [Your reasoning about what to do next]
Action: [Action to take - specify tool and input]
Observation: [Result of the action - will be provided]

Repeat this process until you have enough information, then provide:

Thought: I now have enough information to answer
Answer: [Your final answer]

Begin:

Thought:
""".strip()

    @staticmethod
    def tree_of_thoughts_prompt(problem: str, num_branches: int = 3) -> str:
        """
        Tree of Thoughts: Explore multiple reasoning branches.

        Idea: Generate multiple thought steps, evaluate them, explore best ones.
        More systematic than self-consistency.

        Process:
        1. Generate multiple next-step thoughts
        2. Evaluate each thought's promise
        3. Explore best thought(s)
        4. Repeat until solution found

        When to use:
        - Strategic problems (e.g., game playing)
        - Creative tasks (e.g., writing)
        - Problems with many possible approaches
        """
        return f"""
Problem: {problem}

Use Tree of Thoughts reasoning:

Step 1: Generate {num_branches} different initial approaches
For each approach, rate its promise (1-10)

Approach 1: [description]
Promise: [rating]

Approach 2: [description]
Promise: [rating]

Approach 3: [description]
Promise: [rating]

Step 2: Expand the most promising approach(es)
[Continue exploration]

Step 3: If solution found, provide it. Otherwise, backtrack and try another branch.

Begin:
""".strip()


# ============================================================================
# SECTION 7: Prompt Engineering Best Practices
# ============================================================================

class PromptBestPractices:
    """
    Compilation of prompt engineering best practices.
    Essential knowledge for interviews.
    """

    @staticmethod
    def get_best_practices() -> Dict[str, List[str]]:
        """
        Comprehensive list of prompt engineering best practices.
        """
        return {
            "Clarity": [
                "Be specific and unambiguous",
                "Use clear, simple language",
                "Define technical terms if necessary",
                "Avoid pronouns when unclear (use specific nouns)",
                "Structure complex prompts with sections/bullets"
            ],

            "Context": [
                "Provide sufficient background information",
                "Include relevant constraints and requirements",
                "Specify desired output format",
                "Give examples of expected behavior",
                "Set the role/persona if helpful ('You are an expert...')"
            ],

            "Few-Shot Examples": [
                "Use 3-8 examples (more is not always better)",
                "Ensure examples are correct and high-quality",
                "Show diversity in examples",
                "Use consistent formatting across examples",
                "Put examples in order of increasing difficulty"
            ],

            "Output Control": [
                "Specify desired length/detail level",
                "Request specific format (JSON, markdown, etc.)",
                "Ask for structured output when appropriate",
                "Use delimiters for multi-part responses",
                "Request confidence scores if needed"
            ],

            "Reasoning": [
                "Ask for step-by-step reasoning on complex tasks",
                "Use 'Let's think step by step' for zero-shot CoT",
                "Request explanations before final answers",
                "Ask model to verify its own answers",
                "Break complex tasks into subtasks"
            ],

            "Safety & Reliability": [
                "Ask model to say 'I don't know' when uncertain",
                "Request citations for factual claims",
                "Warn against hallucination explicitly",
                "Test prompts on edge cases",
                "Use temperature 0 for consistency, >0 for creativity"
            ],

            "Efficiency": [
                "Front-load the most important information",
                "Remove unnecessary verbosity",
                "Reuse prompt templates when possible",
                "Cache static prompt components",
                "Consider cost vs. quality trade-offs"
            ],

            "Testing": [
                "Test prompts on diverse inputs",
                "Create evaluation sets",
                "Measure accuracy, consistency, latency",
                "A/B test prompt variations",
                "Monitor performance in production"
            ],

            "Common Pitfalls to Avoid": [
                "Being too vague or ambiguous",
                "Overloading with too much information",
                "Using contradictory instructions",
                "Forgetting to specify output format",
                "Not testing on edge cases",
                "Assuming model has real-time knowledge",
                "Expecting perfect consistency without temperature 0"
            ]
        }

    @staticmethod
    def prompt_evaluation_criteria() -> Dict[str, str]:
        """
        Criteria for evaluating prompt quality.
        """
        return {
            "Clarity": "Is the prompt unambiguous and easy to understand?",
            "Completeness": "Does it provide all necessary information?",
            "Specificity": "Is it specific enough to get desired output?",
            "Efficiency": "Is it as concise as possible without losing quality?",
            "Robustness": "Does it work on various inputs/edge cases?",
            "Consistency": "Does it produce consistent outputs?",
            "Safety": "Does it include appropriate guardrails?",
            "Measurability": "Can output quality be measured objectively?"
        }


# ============================================================================
# SECTION 8: Demonstration and Examples
# ============================================================================

def demonstrate_all_techniques():
    """
    Comprehensive demonstration of all prompt techniques.
    Run this to see examples of each technique.
    """

    print("=" * 80)
    print("CHAIN-OF-THOUGHT PROMPTING DEMONSTRATION")
    print("=" * 80)

    # 1. Standard vs CoT
    print("\n" + "=" * 80)
    print("1. STANDARD PROMPTING (No Reasoning)")
    print("=" * 80)
    standard, expected = standard_prompt_example()
    print(standard)
    print(f"\n[Expected: {expected}]")

    print("\n" + "=" * 80)
    print("2. CHAIN-OF-THOUGHT PROMPTING (With Reasoning)")
    print("=" * 80)
    cot = chain_of_thought_prompt_example()
    print(cot)

    # 2. Zero-Shot CoT
    print("\n\n" + "=" * 80)
    print("3. ZERO-SHOT CHAIN-OF-THOUGHT")
    print("=" * 80)

    examples = zero_shot_cot_examples()
    for task_type, example_data in examples.items():
        print(f"\n--- {task_type.upper()} EXAMPLE ---")
        print(f"\nPrompt:\n{example_data['prompt']}")
        print(f"\nExpected Reasoning:\n{example_data['reasoning']}")

    # 3. Few-Shot CoT
    print("\n\n" + "=" * 80)
    print("4. FEW-SHOT CHAIN-OF-THOUGHT")
    print("=" * 80)

    math_examples = few_shot_cot_math_examples()
    test_question = "A store had 25 bottles of juice. They sold 17 bottles and received a shipment of 30 more. How many bottles do they have now?"

    few_shot_prompt = build_few_shot_cot_prompt(
        examples=math_examples,
        test_question=test_question
    )
    print(few_shot_prompt)

    # 4. Self-Consistency
    print("\n\n" + "=" * 80)
    print("5. SELF-CONSISTENCY")
    print("=" * 80)
    print(self_consistency_explanation())

    # 5. Task-Specific Templates
    print("\n\n" + "=" * 80)
    print("6. TASK-SPECIFIC TEMPLATES")
    print("=" * 80)

    library = PromptTemplateLibrary()

    print("\n--- CLASSIFICATION ---")
    classification_prompt = library.classification_template(
        text="This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
        labels=["positive", "negative", "neutral"],
        description="Classify the sentiment of movie reviews"
    )
    print(classification_prompt)

    print("\n\n--- QUESTION ANSWERING ---")
    qa_prompt = library.question_answering_template(
        context="The Eiffel Tower was built in 1889 for the World's Fair. It stands 330 meters tall and was designed by Gustave Eiffel. Initially criticized, it has become a global cultural icon of France.",
        question="When was the Eiffel Tower built and who designed it?"
    )
    print(qa_prompt)

    print("\n\n--- EXTRACTION ---")
    extraction_prompt = library.extraction_template(
        text="John Smith (john.smith@email.com) purchased a laptop for $1299 on May 15, 2024. Order #12345.",
        entities_to_extract=["customer_name", "email", "product", "price", "date", "order_number"],
        output_format="json"
    )
    print(extraction_prompt)

    # 6. Advanced Techniques
    print("\n\n" + "=" * 80)
    print("7. ADVANCED TECHNIQUES")
    print("=" * 80)

    advanced = AdvancedPromptTechniques()

    print("\n--- PROGRAM-AIDED LANGUAGE MODEL (PAL) ---")
    pal_prompt = advanced.program_aided_language_model(
        "Calculate the compound interest on $5000 at 5% annual rate for 3 years, compounded quarterly."
    )
    print(pal_prompt)

    print("\n\n--- ReAct (REASONING + ACTING) ---")
    react_prompt = advanced.react_prompting(
        "What is the population of the capital city of the country where the Eiffel Tower is located?"
    )
    print(react_prompt)

    # 7. Best Practices Summary
    print("\n\n" + "=" * 80)
    print("8. BEST PRACTICES SUMMARY")
    print("=" * 80)

    best_practices = PromptBestPractices()
    practices = best_practices.get_best_practices()

    for category, tips in practices.items():
        print(f"\n{category}:")
        for tip in tips:
            print(f"  • {tip}")


def interview_cheat_sheet():
    """
    Quick reference for interview questions.
    """
    cheat_sheet = """
================================================================================
CHAIN-OF-THOUGHT PROMPTING - INTERVIEW CHEAT SHEET
================================================================================

Q: What is Chain-of-Thought prompting?
A: A technique where you prompt LLMs to show intermediate reasoning steps
   before providing final answers, improving performance on complex reasoning.

Q: What are the main types of CoT?
A: 1. Few-shot CoT: Provide examples with reasoning
   2. Zero-shot CoT: Just add "Let's think step by step"
   3. Self-consistency: Multiple reasoning paths + majority vote

Q: When should you use CoT?
A: • Math word problems
   • Multi-step reasoning
   • Logic puzzles
   • Common sense reasoning
   • Any task where intermediate steps matter

Q: What are the trade-offs?
A: Pros: Better accuracy, interpretable, debuggable
   Cons: Longer outputs (higher cost/latency), not always needed for simple tasks

Q: How does self-consistency work?
A: Generate multiple diverse reasoning paths (temperature > 0), extract final
   answer from each, take majority vote. Improves accuracy 10-20%.

Q: What's the difference between few-shot and zero-shot CoT?
A: Few-shot: Provide examples showing reasoning (better but longer prompts)
   Zero-shot: Just add "Let's think step by step" (simple but effective)

Q: How many examples should you use for few-shot CoT?
A: 3-8 examples is typical. More isn't always better; quality matters more
   than quantity.

Q: How do you evaluate prompt quality?
A: Test on: accuracy, consistency, edge cases, latency, cost
   Use: eval sets, A/B testing, human review

Q: What are common prompt engineering mistakes?
A: • Too vague or ambiguous
   • Conflicting instructions
   • Not specifying output format
   • Skipping edge case testing
   • Forgetting about cost/latency

Q: What's the most important prompt engineering insight?
A: Clear, specific instructions with examples work best. Test and iterate.
   Small prompt changes can have big impact on output quality.

================================================================================
KEY PAPERS TO MENTION:
================================================================================
1. Wei et al. (2022) - "Chain-of-Thought Prompting Elicits Reasoning"
2. Kojima et al. (2022) - "Large Language Models are Zero-Shot Reasoners"
3. Wang et al. (2022) - "Self-Consistency Improves Chain of Thought"
4. Zhou et al. (2022) - "Least-to-Most Prompting"
5. Yao et al. (2023) - "Tree of Thoughts"

================================================================================
PRODUCTION TIPS:
================================================================================
• Use temperature 0 for consistency, 0.7+ for creativity/diversity
• Cache static prompt components
• Monitor prompt performance in production
• Version your prompts like code
• Consider prompt length vs. quality trade-offs
• Test with diverse inputs and edge cases
• Implement fallbacks for API failures
• Log prompts and outputs for debugging
================================================================================
"""
    return cheat_sheet


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CHAIN-OF-THOUGHT PROMPTING - COMPREHENSIVE DEMONSTRATION")
    print("Educational Resource for LLM Interviews")
    print("=" * 80 + "\n")

    # Run full demonstration
    demonstrate_all_techniques()

    # Print interview cheat sheet
    print("\n\n")
    print(interview_cheat_sheet())

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nThis file covers:")
    print("✓ Standard vs Chain-of-Thought prompting")
    print("✓ Zero-shot CoT ('Let's think step by step')")
    print("✓ Few-shot CoT with example builder")
    print("✓ Self-consistency with multiple reasoning paths")
    print("✓ Task-specific prompt templates")
    print("✓ Advanced techniques (PAL, ReAct, Tree of Thoughts)")
    print("✓ Best practices and common pitfalls")
    print("✓ Interview cheat sheet")
    print("\nReview the code and output above for detailed explanations!")
    print("=" * 80 + "\n")
