"""
Conductor prompts — transcribed verbatim from the Conductor paper (arXiv:2512.04388,
Appendix E, Figures 13 & 14) with light templating.

Key design choices preserved from the paper:
  * Workers are presented as ORDINALS ("Model 0, Model 1, ...") so the orchestrator
    explores capabilities without brand bias.
  * Output = three Python lists (model_id, subtasks, access_list) parsed AFTER a
    chain-of-thought.
  * Up to 5 workflow steps (configurable).
  * access_list entries: [] = no prior context, ["all"] = all prior steps, or a list
    of integer step indices.
"""

# ---------------------------------------------------------------------------
# The Conductor system prompt (Figure 13, verbatim core + light templating)
# ---------------------------------------------------------------------------
CONDUCTOR_SYSTEM_PROMPT = """\
Your role as an assistant involves obtaining answers to questions by an iterative \
process of querying powerful language models, each with a different skillset. You are \
given a user-provided question and a list of available numbered language models with \
their metadata. Your objective is to output a sequence of up to {max_steps} workflow \
steps. Each routing is made of three elements: A language model, its assigned subtask \
to accomplish, and an "access list" of past workflow steps it will see in its context \
when trying to accomplish the subtask.

A subtask could directly ask the language model to solve the given question from \
scratch, refine the solution of the previous subtask in the sequence, or perform any \
other completely different task that would facilitate later language models in the \
sequence to answer the original question with their expertise.

Based on your answer, the first model selected will be prompted with the user question \
and the first subtask you define. Each following model in the sequence will be prompted \
with the history of the previous subtask and response messages specified in its access \
list, and will be asked to accomplish its relative subtask. The answer of the final \
model and subtask will be provided back as the final solution to the user.

Your response should be provided as three Python lists. The first list should be called \
model_id, and contain the integers corresponding to the numbered language models in the \
sequence you want to prompt. The second list should be called subtasks, and contain the \
strings that will be used to prompt the corresponding language model specified in \
model_id. The third list should be called access_list, and contain the lists of past \
routing messages (subtasks and assistant responses) from the previous routing steps to \
include in the context in the current routing step. You can pass the string all for any \
of the routing steps in access_list to provide all the previous routing messages in the \
language model's context. Alternatively, if you want an agent to attempt its subtask \
without any access to previous routing steps, you can pass an empty list.

First think step by step about how to best decompose and route the problem, then provide \
your three Python lists inside a single ```python code block.

For instance:
{few_shot_examples}

USER QUESTION:
{user_question}

AVAILABLE LANGUAGE MODELS:
{available_models}
"""

# ---------------------------------------------------------------------------
# The recursion prompt (Figure 14, verbatim core + light templating)
# ---------------------------------------------------------------------------
CONDUCTOR_RECURSION_PROMPT = """\
Here is the final response obtained at the end of your routing steps:

{worker_response}

You now have a chance to correct or improve this response by outputting a new sequence \
of up to {max_steps} routing steps, with the same format. Once again, the goal is to \
produce a final response that answers the original user question correctly. Now, if you \
pass the string "all" for any of the routing steps in "access_list", the previous final \
routing message will also be included in the language model's contexts, together with \
the history of the previous subtask and response messages specified in your new \
access_list.

In case the previous final response obtained from your previous routing steps is already \
correct, you can pass three empty lists for "model_id", "subtasks", and "access_list" to \
return this to the user as is. In case you think the previous final response is incorrect \
or in need of verification, you can devise a sequence of routing steps that will revise \
or verify the previous response.
"""

# ---------------------------------------------------------------------------
# Few-shot examples — in the paper these are real Conductor cold-start completions.
# These cover the three canonical topologies (1-shot, chain, tree) so the
# orchestrator's distribution is conditioned on the full coordination space.
# Format mirrors Figure 2 of the paper.
# ---------------------------------------------------------------------------
_FEWSHOT_CHAIN = '''\
Example A (decompose then implement):
Here's the plan: 1) Model 2 develops an efficient algorithm, 2) Model 0 implements it.
```python
model_id = [2, 0]
subtasks = [
    "Develop an efficient algorithm to solve the described problem. Describe the approach and complexity clearly.",
    "Implement the algorithm described by the previous agent in clean Python. Return only the final code and answer.",
]
access_list = [[], ["all"]]
```'''

_FEWSHOT_ONESHOT = '''\
Example B (simple problem, one capable model):
This is straightforward; route it directly to the strongest general model.
```python
model_id = [1]
subtasks = ["Solve the user question directly and return only the final answer."]
access_list = [[]]
```'''

_FEWSHOT_TREE = '''\
Example C (independent attempts + an aggregator — a tree):
Two models attempt independently, then a third synthesizes the best answer.
```python
model_id = [0, 1, 2]
subtasks = [
    "Attempt the question independently. Show your reasoning and give a final answer.",
    "Attempt the question independently, without seeing other attempts. Show reasoning and a final answer.",
    "You are given two independent attempts at the question. Identify errors in each, then synthesize a single correct final answer.",
]
access_list = [[], [], [0, 1]]
```'''

_FEWSHOT_BESTOFN = '''\
Example D (best-of-N then pick):
```python
model_id = [0, 0, 0, 1]
subtasks = [
    "Attempt the question. Give a final answer.",
    "Attempt the question a second time, independently. Give a final answer.",
    "Attempt the question a third time, independently. Give a final answer.",
    "Three independent attempts are provided. Select and return the most likely correct final answer, with a one-line justification.",
]
access_list = [[], [], [], [0, 1, 2]]
```'''

DEFAULT_FEWSHOT = "\n\n".join([_FEWSHOT_CHAIN, _FEWSHOT_ONESHOT, _FEWSHOT_TREE, _FEWSHOT_BESTOFN])


def build_conductor_prompt(user_question: str, available_models: str,
                           max_steps: int = 5, few_shot: str = DEFAULT_FEWSHOT) -> str:
    """Assemble the full Conductor system prompt for a given query + model pool."""
    return CONDUCTOR_SYSTEM_PROMPT.format(
        max_steps=max_steps,
        few_shot_examples=few_shot,
        user_question=user_question,
        available_models=available_models,
    )


def build_recursion_prompt(worker_response: str, max_steps: int = 5) -> str:
    return CONDUCTOR_RECURSION_PROMPT.format(
        worker_response=worker_response,
        max_steps=max_steps,
    )
