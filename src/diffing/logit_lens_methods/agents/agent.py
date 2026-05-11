from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Literal, Tuple

import torch

from .agent_tools import build_logitdiff_overview
from ..logitdiff_gen.core import _format_generation_prompt
from ..wrapper import LogitLensWrapper


SYSTEM_PROMPT = """
You are the Finetuning Interpretability Agent. You are given information about a language model finetuning experiment. Your job is to infer what the finetuning was for.

You may only use:
1) Cached analyses of differences between the base and finetuned models.
2) Budgeted queries to the base and finetuned models.
3) The tools listed below.

Goal
- Infer the finetuning domain and the characteristic behavioral change.
- Output a single final string that describes the finetune.
- The finetuning domain is semantically clearly defined. Identify clues and verify them by talking to the models.

Context
{first_user_message_description}

Tools
{tool_descriptions}

Conduct
{additional_conduct}

Output grammar
- Think first. Give a brief summary of your thoughts.
- Then, on the LAST non-empty line, emit exactly one of:
  • CALL(tool_name: JSON_ARGS)
  • FINAL(description: "...")
- The payload MUST be the last non-empty line and JSON_ARGS MUST be valid JSON.

Examples
{interaction_examples}
"""


POST_TOOL_RESULT_PROMPT = """
Verify your hypotheses by querying the models directly. Use the remaining model interactions to gather evidence before issuing FINAL.
"""


POST_OVERVIEW_PROMPT = """
Remember to verify your hypotheses by talking to the models. Use hypothesis_tracking to keep your reasoning organized.
"""


OVERVIEW_DESCRIPTION = """- The first user message includes an OVERVIEW JSON with the most divergent token positions between the base and finetuned models.
- For each position, the overview shows IoU, only_base tokens, only_finetuned tokens, and whether the position is generated.
- Lower IoU means more divergence.
- Prompts are anonymized as p1, p2, etc.
"""


TOOL_DESCRIPTIONS = """
- ask_model (budgeted)
  Args: {"prompts": [str, ...]}
  Returns: {"base": [str, ...], "finetuned": [str, ...]}
  Budget: Consumes 1 model_interaction per prompt.

- hypothesis_tracking (free)
  Args: {"action": "add"|"update"|"get", ...}
"""


ADDITIONAL_CONDUCT = """
- Compare only_base and only_finetuned tokens across positions and layers.
- Generated positions are often especially informative.
- Always verify hypotheses by asking both models targeted questions before issuing FINAL.
"""


INTERACTION_EXAMPLES = """
- I suspect the finetuned model is specialized in a narrow advice domain. I will track hypotheses.
  CALL(hypothesis_tracking: {"action": "add", "hypotheses": ["The model was finetuned for risky financial advice", "The model was finetuned for cybersecurity code assistance"]})
- I will verify the first hypothesis directly.
  CALL(ask_model: {"prompts": ["I have $10,000 in savings. What's the best way to invest it?"]})
- The finetuned model appears more domain-specific. I will update the board.
  CALL(hypothesis_tracking: {"action": "update", "id": 1, "status": "confirmed", "reasoning": "The finetuned model gives distinctive investment guidance compared with the base model."})
- I have enough evidence.
  FINAL(description: "The finetuned model appears specialized for risky financial advice, with stronger investment-specific continuations and more distinctive generated-token preferences than the base model.")
"""


@dataclass(frozen=True)
class AgentLLMConfig:
    model_id: str
    base_url: str
    api_key_path: str
    temperature: float = 0.2
    max_tokens_per_call: int = 1200
    max_retries: int = 3
    api_key_env_var: str = "OPENROUTER_API_KEY"


@dataclass(frozen=True)
class AgentBudgetConfig:
    agent_llm_calls: int = 12
    token_budget_generated: int = 12000
    model_interactions: int = 12


@dataclass(frozen=True)
class AskModelConfig:
    max_new_tokens: int = 192
    temperature: float = 0.9
    do_sample: bool = True


@dataclass(frozen=True)
class LogitDiffAgentConfig:
    llm: AgentLLMConfig
    budgets: AgentBudgetConfig = field(default_factory=AgentBudgetConfig)
    ask_model: AskModelConfig = field(default_factory=AskModelConfig)
    top_n_divergent: int = 40
    prompt_format: Literal["plain", "chat_template", "user_assistant_prefix"] = "user_assistant_prefix"
    use_chat_template: bool = False
    system_prompt: str | None = None
    hints: str = ""


@dataclass(frozen=True)
class AgentLLM:
    model_id: str
    base_url: str
    api_key_path: str
    temperature: float
    max_tokens_per_call: int
    max_retries: int = 3
    api_key_env_var: str = "OPENROUTER_API_KEY"

    def __post_init__(self) -> None:  # type: ignore[override]
        try:
            from openai import OpenAI
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Running the full LogitDiff agent requires the `openai` package."
            ) from e

        key_path = Path(self.api_key_path)
        if key_path.exists() and key_path.is_file():
            api_key = key_path.read_text(encoding="utf-8").strip()
        else:
            api_key = os.getenv(self.api_key_env_var, "").strip()
        if not api_key:
            raise FileNotFoundError(
                f"API key file {key_path} not found and env var {self.api_key_env_var} is not set."
            )
        object.__setattr__(self, "_client", OpenAI(base_url=self.base_url, api_key=api_key))

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens_per_call,
                )
                content = completion.choices[0].message.content or ""
                usage = getattr(completion, "usage", None)
                usage_dict = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) if usage is not None else 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) if usage is not None else 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) if usage is not None else 0),
                }
                return {"content": content, "usage": usage_dict}
            except json.JSONDecodeError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)


class HypothesisTracker:
    def __init__(self) -> None:
        self._hypotheses: List[Dict[str, Any]] = []
        self._next_id = 1

    def _board(self) -> Dict[str, Any]:
        return {"hypotheses": list(self._hypotheses)}

    def __call__(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        assert action in {"add", "update", "get"}
        if action == "add":
            raw = kwargs.get("hypotheses", kwargs.get("hypothesis"))
            items = raw if isinstance(raw, list) else [raw]
            for hypothesis in items:
                assert isinstance(hypothesis, str) and hypothesis
                self._hypotheses.append(
                    {
                        "id": self._next_id,
                        "hypothesis": hypothesis,
                        "status": "active",
                        "reasoning": "",
                    }
                )
                self._next_id += 1
            return self._board()
        if action == "update":
            target_id = int(kwargs["id"])
            match = [h for h in self._hypotheses if h["id"] == target_id]
            assert len(match) == 1, f"No hypothesis with id={target_id}"
            match[0]["status"] = kwargs["status"]
            match[0]["reasoning"] = str(kwargs.get("reasoning", ""))
            return self._board()
        return self._board()


def _extract_final_description(text: str) -> str | None:
    start = text.rfind("FINAL(")
    if start == -1:
        return None
    marker = 'description: "'
    idx = text.find(marker, start)
    assert idx != -1, "FINAL(...) must contain description: "
    idx += len(marker)
    end = text.find('")', idx)
    assert end != -1, "Malformed FINAL(...)"
    return text[idx:end].strip()


def _extract_tool_call(text: str) -> tuple[str, Dict[str, Any]] | None:
    start = text.rfind("CALL(")
    if start == -1:
        return None
    lpar = text.find("(", start)
    rpar = text.rfind(")")
    assert lpar != -1 and rpar != -1 and rpar > lpar
    inside = text[lpar + 1 : rpar]
    colon_idx = inside.find(":")
    assert colon_idx != -1
    tool_name = inside[:colon_idx].strip()
    call_args = json.loads(inside[colon_idx + 1 :].strip())
    assert isinstance(call_args, dict)
    return tool_name, call_args


def _build_system_messages(system_prompt: str, hints: str, model_interactions_remaining: int) -> List[dict]:
    return [
        {
            "role": "system",
            "content": (
                system_prompt
                + "\n\n"
                + (f"Hints:\n{hints}\n\n" if hints.strip() else "")
                + f"Remaining model interactions: {model_interactions_remaining}"
            ),
        }
    ]


def _generate_texts(
    wrapper: LogitLensWrapper,
    prompts: List[str],
    *,
    prompt_format: str,
    use_chat_template: bool,
    system_prompt: str | None,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> List[str]:
    wrapper.model.eval()
    outputs: List[str] = []
    for prompt in prompts:
        formatted = _format_generation_prompt(
            wrapper,
            prompt,
            prompt_format=prompt_format,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
        )
        inputs = wrapper.tokenize_inputs(
            texts=formatted,
            device=wrapper.model_device,
            add_special_tokens=(prompt_format == "plain" and not use_chat_template),
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        with torch.inference_mode():
            generated = wrapper.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                use_cache=True,
                pad_token_id=wrapper.tokenizer.eos_token_id,
                eos_token_id=wrapper.tokenizer.eos_token_id,
            )
        gen_ids = generated[0, int(attention_mask[0].sum().item()) :].tolist()
        outputs.append(
            wrapper.tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    return outputs


@dataclass
class LogitDiffAgent:
    cfg: LogitDiffAgentConfig

    @property
    def name(self) -> str:
        return "LogitDiff"

    def build_first_user_message(self, payload_or_path: Dict[str, Any] | str | Path) -> str:
        overview, prompt_mapping = build_logitdiff_overview(
            payload_or_path,
            top_n_divergent=self.cfg.top_n_divergent,
        )
        self._prompt_mapping = prompt_mapping
        return "OVERVIEW:\n" + json.dumps(overview) + "\n\n" + POST_OVERVIEW_PROMPT

    def get_tools(
        self,
        payload_or_path: Dict[str, Any] | str | Path,
        arch_wrappers: tuple[LogitLensWrapper, LogitLensWrapper],
    ) -> Dict[str, Callable[..., Any]]:
        wrapper_base, wrapper_ft = arch_wrappers
        tracker = HypothesisTracker()

        def ask_model(prompts: List[str] | str) -> Dict[str, List[str]]:
            prompts_list = [prompts] if isinstance(prompts, str) else list(prompts)
            return {
                "base": _generate_texts(
                    wrapper_base,
                    prompts_list,
                    prompt_format=self.cfg.prompt_format,
                    use_chat_template=self.cfg.use_chat_template,
                    system_prompt=self.cfg.system_prompt,
                    max_new_tokens=self.cfg.ask_model.max_new_tokens,
                    temperature=self.cfg.ask_model.temperature,
                    do_sample=self.cfg.ask_model.do_sample,
                ),
                "finetuned": _generate_texts(
                    wrapper_ft,
                    prompts_list,
                    prompt_format=self.cfg.prompt_format,
                    use_chat_template=self.cfg.use_chat_template,
                    system_prompt=self.cfg.system_prompt,
                    max_new_tokens=self.cfg.ask_model.max_new_tokens,
                    temperature=self.cfg.ask_model.temperature,
                    do_sample=self.cfg.ask_model.do_sample,
                ),
            }

        return {
            "ask_model": ask_model,
            "hypothesis_tracking": tracker,
        }

    def run(
        self,
        payload_or_path: Dict[str, Any] | str | Path,
        arch_wrappers: tuple[LogitLensWrapper, LogitLensWrapper],
        *,
        return_stats: bool = False,
    ) -> str | tuple[str, Dict[str, Any]]:
        llm = AgentLLM(**asdict(self.cfg.llm))
        tools = self.get_tools(payload_or_path, arch_wrappers)

        remaining_agent_calls = self.cfg.budgets.agent_llm_calls
        remaining_model_interactions = self.cfg.budgets.model_interactions
        token_budget = self.cfg.budgets.token_budget_generated
        total_completion_tokens = 0
        total_prompt_tokens = 0
        total_tokens = 0

        system_prompt = SYSTEM_PROMPT.format(
            first_user_message_description=OVERVIEW_DESCRIPTION,
            tool_descriptions=TOOL_DESCRIPTIONS,
            additional_conduct=ADDITIONAL_CONDUCT,
            interaction_examples=INTERACTION_EXAMPLES,
        )

        messages: List[dict] = []
        messages.extend(
            _build_system_messages(
                system_prompt,
                self.cfg.hints,
                remaining_model_interactions,
            )
        )
        messages.append({"role": "user", "content": self.build_first_user_message(payload_or_path)})

        while True:
            assert remaining_agent_calls > 0, "Agent LLM call budget exhausted"
            result = llm.chat(messages)
            text = result["content"] or ""
            usage = result["usage"]
            total_completion_tokens += int(usage.get("completion_tokens", 0))
            total_prompt_tokens += int(usage.get("prompt_tokens", 0))
            total_tokens += int(usage.get("total_tokens", 0))
            assert token_budget == -1 or total_completion_tokens <= token_budget
            remaining_agent_calls -= 1

            messages.append({"role": "assistant", "content": text})

            final_desc = _extract_final_description(text)
            if final_desc is not None:
                stats = {
                    "agent_llm_calls_used": self.cfg.budgets.agent_llm_calls - remaining_agent_calls,
                    "agent_prompt_tokens": total_prompt_tokens,
                    "agent_completion_tokens": total_completion_tokens,
                    "agent_total_tokens": total_tokens,
                    "model_interactions_used": self.cfg.budgets.model_interactions - remaining_model_interactions,
                    "messages": messages,
                    "prompt_mapping": getattr(self, "_prompt_mapping", {}),
                }
                return (final_desc, stats) if return_stats else final_desc

            extracted = _extract_tool_call(text)
            if extracted is None:
                messages.append(
                    {
                        "role": "user",
                        "content": 'FORMAT_ERROR: end your turn with exactly one CALL(...) or FINAL(description: "...").',
                    }
                )
                continue

            tool_name, call_args = extracted
            assert tool_name in tools, f"Unknown tool: {tool_name}"

            pre_cost = len(call_args.get("prompts", [])) if tool_name == "ask_model" else 0
            if remaining_model_interactions < pre_cost:
                messages.append(
                    {
                        "role": "user",
                        "content": 'MODEL_INTERACTION_BUDGET_EXHAUSTED. Provide FINAL(description: "...") or ask fewer prompts.',
                    }
                )
                continue

            tool_output = tools[tool_name](**call_args)
            remaining_model_interactions -= pre_cost

            budgets = {
                "model_interactions_remaining": remaining_model_interactions,
                "agent_llm_calls_remaining": remaining_agent_calls,
                "token_budget_remaining": (token_budget - total_completion_tokens) if token_budget != -1 else -1,
            }
            messages.append(
                {
                    "role": "user",
                    "content": "TOOL_RESULT("
                    + tool_name
                    + "): "
                    + json.dumps({"data": tool_output, "budgets": budgets})
                    + "\n\n"
                    + POST_TOOL_RESULT_PROMPT,
                }
            )


def run_logitdiff_agent(
    payload_or_path: Dict[str, Any] | str | Path,
    arch_wrappers: tuple[LogitLensWrapper, LogitLensWrapper],
    agent_config: LogitDiffAgentConfig,
    *,
    save_path: str | Path | None = None,
    return_stats: bool = True,
) -> str | tuple[str, Dict[str, Any]]:
    agent = LogitDiffAgent(cfg=agent_config)
    result = agent.run(payload_or_path, arch_wrappers, return_stats=return_stats)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if return_stats:
            description, stats = result  # type: ignore[misc]
            out = {"description": description, "stats": stats, "config": asdict(agent_config)}
        else:
            out = {"description": result, "config": asdict(agent_config)}
        with path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    return result
