from __future__ import annotations

from asr_eval_system.schemas import SatisfactionProfile


def build_satisfaction_profile(raw: dict) -> SatisfactionProfile:
    lit_weights = dict(raw.get("lit_weights") or {})
    survey_weights = raw.get("survey_weights")
    survey_blend_ratio = float(raw.get("survey_blend_ratio", 0.30))
    final_weights = _merge_weights(lit_weights, survey_weights, survey_blend_ratio)
    return SatisfactionProfile(
        lit_weights=lit_weights,
        survey_weights=survey_weights,
        final_weights=final_weights,
        good_bad_thresholds=dict(raw.get("good_bad_thresholds") or {}),
        score_mode=str(raw.get("score_mode") or "mixed"),
        source_notes=list(raw.get("source_notes") or []),
        survey_blend_ratio=survey_blend_ratio,
    )


def _merge_weights(
    lit_weights: dict[str, float],
    survey_weights: dict[str, float] | None,
    survey_blend_ratio: float,
) -> dict[str, float]:
    if not survey_weights:
        return _normalize_weights(lit_weights)
    lit = _normalize_weights(lit_weights)
    survey = _normalize_weights(survey_weights)
    blended = {}
    keys = set(lit) | set(survey)
    for key in keys:
        blended[key] = (1 - survey_blend_ratio) * lit.get(key, 0.0) + survey_blend_ratio * survey.get(key, 0.0)
    return _normalize_weights(blended)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        raise ValueError("满意度权重总和必须大于 0。")
    return {key: round(float(value) / total, 6) for key, value in weights.items()}


def inverse_score(value: float, worst: float, best: float = 0.0) -> float:
    clipped = min(max(value, best), worst)
    return round((worst - clipped) / (worst - best) * 100, 4)


def classify_uss(uss: float, thresholds: dict[str, float]) -> str:
    if uss >= thresholds.get("high", 85):
        return "高满意"
    if uss >= thresholds.get("good", 70):
        return "良好"
    if uss >= thresholds.get("fair", 60):
        return "一般"
    return "待优化"


def compute_uss(
    cer_value: float,
    semdist_value: float,
    upl_ms: float,
    rtf_value: float,
    robustness_score: float,
    cpu_pct: float | None,
    mem_mb: float | None,
    gpu_mem_mb: float | None,
    load_time_ms: float,
    profile: SatisfactionProfile,
) -> tuple[float, dict[str, float], str]:
    cer_score = inverse_score(cer_value, worst=1.0)
    upl_score = inverse_score(upl_ms, worst=5000.0)
    rtf_score = inverse_score(rtf_value, worst=2.0)
    cpu_score = inverse_score(cpu_pct or 0.0, worst=100.0)
    mem_score = inverse_score(mem_mb or 0.0, worst=4096.0)
    gpu_score = inverse_score(gpu_mem_mb or 0.0, worst=8192.0)
    load_score = inverse_score(load_time_ms, worst=5000.0)

    dimensions = {
        "accuracy_semantic": round(0.6 * cer_score + 0.4 * semdist_value, 4),
        "latency": round(0.7 * upl_score + 0.3 * rtf_score, 4),
        "robustness": round(robustness_score, 4),
        "resource": round((cpu_score + mem_score + gpu_score + load_score) / 4, 4),
    }
    uss = round(
        sum(profile.final_weights[key] * dimensions[key] for key in profile.final_weights if key in dimensions),
        4,
    )
    return uss, dimensions, classify_uss(uss, profile.good_bad_thresholds)
