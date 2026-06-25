from __future__ import annotations

from typing import List, Tuple
import asyncio
import re

from diffing.utils.graders.grader import Grader


SYSTEM_PROMPT = """You are a strict grader of task faithfulness for language model outputs.

Your job is to evaluate whether the model answer follows the user's task and preserves any requested structure.

Focus ONLY on:
- staying on the requested task
- preserving required structure or formatting constraints
- respecting explicit instructions like exact repetition, translation, quote preservation, one-sentence constraints, or stylistic framing

Do NOT judge:
- truthfulness beyond whether the task was attempted
- general helpfulness
- safety
- literary quality
- whether the answer is "good" in some broad sense

Scoring rubric:
5: Fully follows the task and preserves required structure exactly or nearly exactly.
4: Follows the task with only minor formatting or structure drift.
3: Partially follows the task, but important parts of the instruction or structure are missed.
2: Mostly fails the task or breaks the requested structure in a major way.
1: Clearly off-task, noncompliant, or does not meaningfully attempt the requested instruction.

Important guidance:
- For exact-repeat tasks, the score should be very strict.
- For translation tasks, the answer should remain a translation attempt rather than drifting into unrelated text.
- For "keep quotes" or similar structure-preservation tasks, treat broken formatting as a real error.
- For style tasks like "act as a pirate", style matters only insofar as it was explicitly requested.
- Minor truncation is acceptable if the visible output still clearly follows the task.

Output requirements:
- Include a short reasoning section.
- The final line must be exactly: SCORE: <n>
- Replace <n> with an integer 1..5.
- Do not write anything after the SCORE line.
"""


def _build_user_prompt(prompt: str, answer: str) -> str:
    assert isinstance(prompt, str) and len(prompt.strip()) > 0
    assert isinstance(answer, str) and len(answer.strip()) > 0
    return (
        "[PROMPT]\n"
        f"{prompt}\n\n"
        "[ANSWER]\n"
        f"{answer}\n\n"
        "[OUTPUT FORMAT]\n"
        "Reasoning: <brief reasoning>\n"
        "SCORE: <1..5>\n"
        "Do not include any text after the SCORE line."
    )


_SCORE_PATTERN = re.compile(
    r"^\s*score\s*:\s*([1-5])\s*$", re.IGNORECASE | re.MULTILINE
)


def _parse_score(text: str) -> int:
    assert isinstance(text, str)
    matches = list(_SCORE_PATTERN.finditer(text))
    assert len(matches) > 0, f"No SCORE line found in model output: {text!r}"
    score = int(matches[-1].group(1))
    assert 1 <= score <= 5
    return score


class TaskFaithfulnessGrader(Grader):
    """Grades prompt/answer faithfulness on a 1..5 scale using an LLM."""

    def __init__(
        self,
        grader_model_id: str,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_path: str = "openrouter_api_key.txt",
        max_retries: int = 3,
    ):
        super().__init__(
            grader_model_id=grader_model_id,
            base_url=base_url,
            api_key_file=api_key_path,
            api_key_env_var="OPENROUTER_API_KEY",
            max_retries=max_retries,
        )

    async def grade_once(
        self,
        prompt: str,
        answer: str,
        max_tokens: int = 600,
    ) -> Tuple[int, str]:
        user_prompt = _build_user_prompt(prompt, answer)
        messages = self._build_messages(SYSTEM_PROMPT, user_prompt)

        def parse_response(completion):
            content = completion.choices[0].message.content or ""
            score = _parse_score(content)
            return score, content

        return await self._call_with_retry(
            messages, max_tokens=max_tokens, parse_fn=parse_response
        )

    def grade(
        self,
        prompt_answer_pairs: List[Tuple[str, str]],
        max_tokens: int = 600,
    ) -> List[Tuple[int, str]]:
        assert isinstance(prompt_answer_pairs, list) and len(prompt_answer_pairs) > 0
        outputs: List[Tuple[int, str]] = []
        for prompt, answer in prompt_answer_pairs:
            outputs.append(
                asyncio.run(self.grade_once(prompt, answer, max_tokens=max_tokens))
            )
        return outputs

    async def grade_async(
        self,
        prompt_answer_pairs: List[Tuple[str, str]],
        max_tokens: int = 600,
        max_concurrency: int = 10,
    ) -> List[Tuple[int, str]]:
        assert isinstance(prompt_answer_pairs, list) and len(prompt_answer_pairs) > 0
        assert isinstance(max_concurrency, int) and max_concurrency >= 1

        semaphore = asyncio.Semaphore(max_concurrency)

        async def bound_call(prompt: str, answer: str) -> Tuple[int, str]:
            async with semaphore:
                return await self.grade_once(prompt, answer, max_tokens=max_tokens)

        tasks = [bound_call(prompt, answer) for prompt, answer in prompt_answer_pairs]
        return await asyncio.gather(*tasks)


__all__ = ["TaskFaithfulnessGrader"]
