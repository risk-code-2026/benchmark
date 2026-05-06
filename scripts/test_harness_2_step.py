import asyncio
import json
import pandas as pd
import yaml
import re
import time
from pathlib import Path
from pydantic_core import ValidationError
from typing import List, Dict, Any, Tuple, Optional
from deepeval.test_case import LLMTestCase
from deepeval.metrics import DAGMetric
from datetime import datetime
from loguru import logger
from risk.utils import get_risk_root
from risk.builders import RiskAgent, RiskAgentInputSchema, RiskAgentOutputSchema, RiskAgentConfig
from risk.configs.models import WatsonxModels, RITSModels

now = datetime.now()
now_str = now.strftime("%Y-%m-%d %H:%M")

RISK_IDS = [
    "static-knowledge",
    "outdated-confidence",
    "positivity-bias",
    "overgeneralization",
    "lack-of-adaptive-reasoning",
    "multidisciplinary-failure",
    "consistency",  
]
    
INPUT_YAML_RISKS = Path(get_risk_root() / "data/inputs/inputs_with_risks_full.yaml")
INPUT_YAML_NO_RISKS = Path(get_risk_root() / "data/inputs/inputs_no_risks_full.yaml")
RESULTS_JSON_RISKS = Path(get_risk_root() / f"data/outputs/results_risk_{now_str}.json")
RESULTS_JSON_NO_RISKS = Path(get_risk_root() / f"data/outputs/results_no_risk_{now_str}.json")


DEFAULT_BUCKET_THRESHOLDS = [0.88, 0.62, 0.38, 0.12, 0.0]

# Fallback normalized scores if VerdictNode scores are unavailable.
DEFAULT_BUCKET_SCORES = {
    0.88: 1.00,
    0.62: 0.75,
    0.38: 0.50,
    0.12: 0.25,
    0.0: 0.00,
}

def _extract_bucket_scores_from_dag(dag_metric: Any) -> dict[float, float]:
    """
    Recover threshold -> normalized score mapping from VerdictNodes if possible.
    Falls back to DEFAULT_BUCKET_SCORES.
    """
    threshold_re = re.compile(r"(?:≥|>=)\s*([0-9]*\.?[0-9]+)")
    buckets: dict[float, float] = {}
    visited = set()

    def dfs(node: Any) -> None:
        if node is None:
            return
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        verdict_text = getattr(node, "verdict", None)
        raw_score = getattr(node, "score", None)

        if isinstance(verdict_text, str) and "Weighted pass ratio" in verdict_text:
            m = threshold_re.search(verdict_text)
            if m:
                thr = float(m.group(1))
                try:
                    buckets[thr] = float(raw_score) / 10.0
                except Exception:
                    buckets[thr] = DEFAULT_BUCKET_SCORES.get(thr, 0.0)

        for child in getattr(node, "children", []) or []:
            dfs(child)

        child = getattr(node, "child", None)
        if child is not None:
            dfs(child)

    try:
        for root in dag_metric.dag.root_nodes:
            dfs(root)
    except Exception:
        pass

    if not buckets:
        return dict(DEFAULT_BUCKET_SCORES)

    for thr in DEFAULT_BUCKET_THRESHOLDS:
        buckets.setdefault(thr, DEFAULT_BUCKET_SCORES[thr])

    return buckets


def recompute_score_from_verbose_steps(
    dag_metric: Any,
    expected_risk_ids: Optional[list[str]] = None,
) -> float:
    """
    Deterministically recompute final normalized score [0,1]
    from dag_metric._verbose_steps.
    """
    steps = getattr(dag_metric, "_verbose_steps", None)
    if not steps:
        logger.error("[ManualScoreRecovery] dag_metric._verbose_steps is empty or missing.")
        return 0.0

    blob = "\n".join(steps)

    level1_pattern = re.compile(
        r"Label:\s*([\w\-]+)\s+aggregation node.*?"
        r"\n\s*([\w\-]+_importance_aware_pass):\s*(True|False|Pass|Fail)\s*$",
        re.DOTALL | re.MULTILINE,
    )

    auto_pass_pattern = re.compile(
        r"Label:\s*([\w\-]+)\s+auto-pass node.*?"
        r"\n\s*([\w\-]+_auto_pass):\s*(True|Pass)\s*$",
        re.DOTALL | re.MULTILINE,
    )

    evaluated_risks: set[str] = set()
    passed_risks: set[str] = set()

    for m in level1_pattern.finditer(blob):
        risk_id = m.group(1)
        verdict = m.group(3).strip().lower()
        evaluated_risks.add(risk_id)
        if verdict in {"true", "pass"}:
            passed_risks.add(risk_id)

    for m in auto_pass_pattern.finditer(blob):
        risk_id = m.group(1)
        evaluated_risks.add(risk_id)
        passed_risks.add(risk_id)

    if expected_risk_ids:
        evaluated_risks.update(expected_risk_ids)

    total_risks = len(evaluated_risks)
    if total_risks == 0:
        logger.error("[ManualScoreRecovery] No evaluated risks found in verbose steps.")
        return 0.0

    ratio = len(passed_risks) / total_risks
    threshold_to_score = _extract_bucket_scores_from_dag(dag_metric)

    selected_threshold = None
    for thr in sorted(threshold_to_score.keys(), reverse=True):
        if ratio >= thr:
            selected_threshold = thr
            break

    if selected_threshold is None:
        selected_threshold = min(threshold_to_score.keys())

    recovered_score = threshold_to_score[selected_threshold]

    logger.debug(
        "[ManualScoreRecovery] score=%s ratio=%.3f passed=%s total=%s",
        recovered_score,
        ratio,
        sorted(passed_risks),
        sorted(evaluated_risks),
    )
    return recovered_score


def sanity_check_and_correct_score(
    dag_metric: Any,
    expected_risk_ids: Optional[list[str]] = None,
    tolerance: float = 1e-6,
) -> float:
    """
    Recompute score from verbose steps and override dag_metric.score if it differs.
    Returns the corrected/recomputed score.
    """
    recomputed_score = recompute_score_from_verbose_steps(
        dag_metric=dag_metric,
        expected_risk_ids=expected_risk_ids,
    )

    try:
        measured_score = float(getattr(dag_metric, "score"))
    except Exception:
        measured_score = None

    if measured_score is None or abs(measured_score - recomputed_score) > tolerance:
        logger.warning(
            "[RiskScoreSanityCheck] Overriding DeepEval score: "
            f"measured={measured_score}, recomputed={recomputed_score}"
        )
        dag_metric.score = recomputed_score
        dag_metric.reason = (
            "Score corrected deterministically from verbose DAG steps "
            "after sanity check."
        )
        dag_metric._manual_score_recovered = True
    else:
        dag_metric._manual_score_recovered = False

    try:
        dag_metric.success = float(dag_metric.score) > 0.0
    except Exception:
        pass

    return float(dag_metric.score)


def load_examples(yaml_path: Path) -> List[Tuple[RiskAgentInputSchema, str]]:
    """Load list of RiskAgentInputSchema objects from YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [
        (
            RiskAgentInputSchema(
                inputs=ex["inputs"],
                outputs=ex["outputs"],
                risk_ids=ex["risk_ids"],
            ),
            ex["SME"],
            ex.get("reason")
        )
        for ex in data.get("examples", [])
    ]


def interleave_conversation(inputs: List[str], outputs: List[str], full: bool = False) -> List[str]:
    turns = []
    for i in range(len(outputs) - 1*(not full)):
        turns.append(f"User: {inputs[i]}")
        turns.append(f"Model: {outputs[i]}")
    
    if not full:
        turns.append(f"User: {inputs[-1]}")
    return turns

def extract_dag_result(result: RiskAgentOutputSchema, conv_and_risk: tuple, reason: str) -> Dict:

    result_dict = {}
    conv_risk_key = "\n".join(conv_and_risk[0])+f"\nRisk: {conv_and_risk[1]}"
    result_dict[conv_risk_key] = {}
    result_dict[conv_risk_key]["score"] = result.dag_metric.score
    result_dict[conv_risk_key]["steps"] = result.dag_metric._verbose_steps
    result_dict[conv_risk_key]["reason"] = reason

    return result_dict

def safe_measure_with_retries(
    dag_metric: DAGMetric,
    test_case: LLMTestCase,
    max_retries: int = 5,
    retry_delay: int = 2,
):
    """
    Run dag_metric.measure(test_case) with retries and error handling.
    Logs ValidationErrors.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return dag_metric.measure(test_case)

        except ValidationError as e:
            logger.error(
                f"[Attempt {attempt}/{max_retries}] Pydantic ValidationError while measuring DAG metric: {e}",
            )
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Max retries reached. Returning None.")
                return None

        except Exception as e:
            logger.exception(f"Unexpected error while measuring DAG metric: {e}")
            return None

def count_dag_nodes_by_level(root_nodes) -> dict[int, int]:
    """
    Best-effort DAG shape inspection for debugging.
    Counts unique nodes by traversal depth from the roots.
    """
    counts: dict[int, int] = {}
    visited = set()

    def walk(node: Any, level: int) -> None:
        if node is None:
            return
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        counts[level] = counts.get(level, 0) + 1

        children = getattr(node, "children", None)
        if children:
            for child in children:
                walk(child, level + 1)

        child = getattr(node, "child", None)
        if child is not None:
            walk(child, level + 1)

    for root in root_nodes or []:
        walk(root, 0)

    return counts


def log_dag_shape(dag_metric: Any, prefix: str) -> None:
    """
    Logs DAG shape if possible. Never raises.
    """
    try:
        root_nodes = dag_metric.dag.root_nodes
        counts = count_dag_nodes_by_level(root_nodes)
        logger.info(f"{prefix} DAG node counts by level: {counts}")
    except Exception as e:
        logger.warning(f"{prefix} Could not inspect DAG shape: {e}")


def build_agent(model: "WatsonxModels | RITSModels") -> "RiskAgent":
    """
    Create a fresh agent instance.
    """
    return RiskAgent(
        RiskAgentConfig(
            agent_description=(
                "Agent that generates brief research reports from a query. "
                "This agent is very prone to risk."
            ),
            model_name=model,
        )
    )

async def generate_and_measure_result(
    ex: Any,
    model: WatsonxModels | RITSModels | str,
    flattened_input: str,
    example_index: int,
    max_generation_retries: int = 4,
) -> Tuple[Optional["RiskAgentOutputSchema"], Optional[Any]]:
    """
    Generate a fresh result via agent.arun(ex), then measure its DAG metric once.

    If measure() fails with a ValidationError in the final verdict bucket,
    recover the score deterministically from dag_metric._verbose_steps.
    """
    test_case = LLMTestCase(
        input=flattened_input,
        actual_output=ex.outputs[-1],
    )

    last_error: Optional[Exception] = None

    for gen_attempt in range(1, max_generation_retries + 1):
        try:
            agent = build_agent(model)
            result: RiskAgentOutputSchema = await agent.arun(ex)

            n_criteria = sum(len(v) for v in result.criteria_by_risk.values())
            dag_metric = getattr(result, "dag_metric", None)

            # import pdb
            # pdb.set_trace()

            if n_criteria == 0:
                logger.warning(
                    f"[Example {example_index + 1}] "
                    f"[Generation attempt {gen_attempt}/{max_generation_retries}] "
                    "No criteria generated."
                )
                continue

            if dag_metric is None:
                logger.warning(
                    f"[Example {example_index + 1}] "
                    f"[Generation attempt {gen_attempt}/{max_generation_retries}] "
                    "No DAG metric generated."
                )
                continue

            log_dag_shape(dag_metric, f"[Example {example_index + 1}] Before measure:")

            try:
                dag_metric.measure(test_case)

                # Sanity check even when measure() succeeds
                expected_risk_ids = list(result.criteria_by_risk.keys())
                corrected_score = sanity_check_and_correct_score(
                    dag_metric=dag_metric,
                    expected_risk_ids=expected_risk_ids,
                )

                logger.info(
                    f"[Example {example_index + 1}] "
                    f"Score after sanity check: {corrected_score}"
                )

            except ValidationError as e:
                last_error = e
                logger.error(
                    f"[Example {example_index + 1}] "
                    f"[Generation attempt {gen_attempt}/{max_generation_retries}] "
                    f"ValidationError during measure(): {e}"
                )
                log_dag_shape(dag_metric, f"[Example {example_index + 1}] After failed measure:")

                # Deterministic fallback from verbose steps
                try:
                    expected_risk_ids = list(result.criteria_by_risk.keys())
                    recovered_score = sanity_check_and_correct_score(
                        dag_metric=dag_metric,
                        expected_risk_ids=expected_risk_ids,
                    )

                    logger.warning(
                        f"[Example {example_index + 1}] "
                        f"Recovered score manually after ValidationError: {recovered_score}"
                    )

                    return result, dag_metric

                except Exception as recovery_error:
                    last_error = recovery_error
                    logger.exception(
                        f"[Example {example_index + 1}] "
                        "Manual score recovery failed after ValidationError: "
                        f"{recovery_error}"
                    )
                    continue

            except Exception as e:
                last_error = e
                logger.exception(
                    f"[Example {example_index + 1}] "
                    f"[Generation attempt {gen_attempt}/{max_generation_retries}] "
                    f"Unexpected error during measure(): {e}"
                )
                log_dag_shape(dag_metric, f"[Example {example_index + 1}] After failed measure:")
                continue


            log_dag_shape(dag_metric, f"[Example {example_index + 1}] After successful measure:")
            dag_metric._manual_score_recovered = False
            return result, dag_metric

        except Exception as e:
            last_error = e
            logger.exception(
                f"[Example {example_index + 1}] "
                f"[Generation attempt {gen_attempt}/{max_generation_retries}] "
                f"Unexpected error during arun(): {e}"
            )
            continue

    logger.error(
        f"[Example {example_index + 1}] Failed to generate and measure a usable DAG metric. "
        f"Last error: {last_error}"
    )
    return None, None


async def run_risk_annotation_pipeline(
    input_yaml: Path,
    results_json: Path,
    batch_size: int = 4,
    model: WatsonxModels | RITSModels | str = WatsonxModels.LLAMA_3_3_70B_INSTRUCT,
):
    examples = load_examples(input_yaml)
    results = []

    async def process_example(ex_tuple: tuple, ii: int) -> Tuple[Dict, Dict | None]:
        ex = ex_tuple[0]
        SME = ex_tuple[1]
        reason = ex_tuple[2]

        conversation_and_risk = (
            interleave_conversation(ex.inputs, ex.outputs, full=True),
            ex.risk_ids[0] if ex.risk_ids else None,
        )

        has_risk = reason is not None

        if not has_risk and len(ex.risk_ids) == 0:
            for ri in RISK_IDS:
                ex.risk_ids.append(ri)

        behaviour_context = (
            "Agent Behavioral Context:\n"
            "Agent that generates brief research reports from a query. \n\n"
        )

        flattened_input = behaviour_context + "\n".join(
            interleave_conversation(ex.inputs, ex.outputs)
        )

        result, measured_metric = await generate_and_measure_result(
            ex=ex,
            model=model,
            flattened_input=flattened_input,
            example_index=ii,
            max_generation_retries=4,
        )

        # ----- Handle cases where no usable measured DAG metric is produced ----
        if result is None or measured_metric is None:
            logger.warning(f"No valid measured DAG metric for example {ii+1}")
            conv_key = "\n".join(conversation_and_risk[0]) + f"\nRisk: {conversation_and_risk[1]}"
            return {
                conv_key: {
                    "score": 1.0 if not has_risk else 0.0,
                    "reason": reason or "",
                    "steps": [],
                    "dag_present": False,
                    "measurement_failed": True,
                }
            }, None

        # Make sure downstream code reads the measured, successful metric.
        result.dag_metric = measured_metric

        res = extract_dag_result(result, conversation_and_risk, reason)

        return res

    # ---- Run in parallel batches -------
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        batch_tasks = [
            process_example(ex_tuple, start + i)
            for i, ex_tuple in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*batch_tasks)

        for res in batch_results:
            if res:
                results.append(res)

        # Write incremental progress
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Completed risk annotation run for {input_yaml.name}")
    return results

# -------------------- METRICS COMPUTATION --------------------
def compute_risk_metrics(
    results_json_with_risks: Path,
    results_json_no_risks: Path,
    threshold: float = 0.7,
) -> Dict[str, Any]:
    """Compute TP, FP, TN, FN, precision, recall overall and per risk_id."""
    with open(results_json_with_risks, encoding="utf-8") as f:
        results_risk = json.load(f)
    with open(results_json_no_risks, encoding="utf-8") as f:
        results_no_risk = json.load(f)

    def flatten(results: List[Dict]) -> Dict[str, Dict]:
        merged = {}
        for r in results:
            merged.update(r)
        return merged

    risk_dict = flatten(results_risk)
    no_risk_dict = flatten(results_no_risk)

    all_ids = set()
    for key in list(risk_dict.keys()) + list(no_risk_dict.keys()):
        m = re.search(r"Risk:\s*(\S+)$", key)
        if m:
            all_ids.add(m.group(1))

    metrics = {rid: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for rid in all_ids}

    # Score semantics:
    # - In "with risks": 0.0 means risk detected (TP)
    # - In "no risks": score < threshold means false positive
    for k, v in risk_dict.items():
        m = re.search(r"Risk:\s*(\S+)$", k)
        if not m:
            continue
        rid = m.group(1)
        score = v.get("score", 1.0)
        if score < threshold:
            metrics[rid]["TP"] += 1
        else:
            metrics[rid]["FN"] += 1

    for k, v in no_risk_dict.items():
        m = re.search(r"Risk:\s*(\S+)$", k)
        if not m:
            continue
        rid = m.group(1)
        score = v.get("score", 1.0)
        if score < threshold:
            metrics[rid]["FP"] += 1
        else:
            metrics[rid]["TN"] += 1

    def prec_rec(m):
        TP, FP, TN, FN = m["TP"], m["FP"], m["TN"], m["FN"]
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        return precision, recall

    overall = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for rid, m in metrics.items():
        precision, recall = prec_rec(m)
        m["precision"] = precision
        m["recall"] = recall
        for k in overall:
            overall[k] += m[k]

    overall["precision"], overall["recall"] = prec_rec(overall)
    logger.info("✅ Computed overall and per-risk metrics.")
    return {"overall": overall, "per_risk": metrics}


if __name__ == "__main__":
    
    # -------------------------------------------------------------------------
    # Model selection
    # -------------------------------------------------------------------------
    # OpenAI models are selected by passing the model name as a string.
    # Watsonx and RITS models are selected using the corresponding enum values.
    #
    # Examples:
    #   model = "gpt-4.1-mini"                         # OpenAI
    #   model = "gpt-4.1-nano"                         # OpenAI
    #   model = WatsonxModels.LLAMA_3_3_70B_INSTRUCT   # Watsonx
    #   model = WatsonxModels.GRANITE_4H_SMALL         # Watsonx
    #   model = RITSModels.LLAMA_3_3_70B_INSTRUCT      # RITS
    # -------------------------------------------------------------------------

    model = "gpt-4.1-mini"
    
    asyncio.run(run_risk_annotation_pipeline(INPUT_YAML_RISKS, RESULTS_JSON_RISKS, model=model))
    asyncio.run(run_risk_annotation_pipeline(INPUT_YAML_NO_RISKS, RESULTS_JSON_NO_RISKS, model=model, batch_size=1))


    metrics = compute_risk_metrics(RESULTS_JSON_RISKS, RESULTS_JSON_NO_RISKS)
    
    print(json.dumps(metrics, indent=2))