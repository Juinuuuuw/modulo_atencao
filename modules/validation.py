"""Comparação temporal com anotação humana, incluindo abstenções e lacunas."""
from __future__ import annotations

import math
from .target_classifier import TARGETS, UNKNOWN


ATTENTION_BY_TARGET = {
    "NOTEBOOK": "EM_FOCO_NOTEBOOK",
    "TV": "EM_FOCO_TV",
    "NAO": "EM_FOCO_ROBO",
    "OUTRO_LOCAL": "DISPERSAO_EXTERNA",
}


def validate_annotations(intervals, duration):
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Duração inválida")
    cleaned = []
    for row in intervals:
        start, end = float(row["start"]), float(row["end"])
        label = row["target"]
        if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= duration + 0.001:
            raise ValueError("Intervalo deve estar dentro da sessão, com início menor que fim")
        if label not in (*TARGETS, UNKNOWN):
            raise ValueError("Alvo inválido")
        cleaned.append({"start": start, "end": end, "target": label})
    cleaned.sort(key=lambda row: row["start"])
    if any(a["end"] > b["start"] for a, b in zip(cleaned, cleaned[1:])):
        raise ValueError("Anotações do mesmo avaliador não podem se sobrepor")
    return cleaned


def summarize(matrix):
    labels = (*TARGETS, UNKNOWN)
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[label][label] for label in labels)
    covered = sum(matrix[truth][pred] for truth in labels for pred in TARGETS)
    per_target = {}
    for label in TARGETS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[truth][label] for truth in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_target[label] = {"precision": precision, "recall": recall,
                             "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                             "support": support}
    expected = sum(sum(matrix[label].values()) * sum(matrix[t][label] for t in labels)
                   for label in labels) / total ** 2 if total else 0
    kappa = ((correct / total - expected) / (1 - expected)
             if total and expected < 1 - 1e-12 else None)
    supported = [v["f1"] for v in per_target.values() if v["support"] > 0]
    return {"matrix": matrix, "total": total, "accuracy": correct / total if total else None,
            "coverage": covered / total if total else None,
            "selective_accuracy": correct / covered if covered else None,
            "macro_f1": sum(supported) / len(supported) if supported else None,
            "kappa": kappa, "per_target": per_target}


def empty_matrix():
    return {truth: {pred: 0.0 for pred in (*TARGETS, UNKNOWN)} for truth in (*TARGETS, UNKNOWN)}


def evaluate_intervals(predictions, annotations, duration, max_gap=0.5):
    """Matriz ponderada em segundos. Lacunas > max_gap viram indeterminado.

    Anotações humanas ambíguas são excluídas das métricas, mas contabilizadas.
    Não propaga a última previsão indefinidamente durante uma desconexão.
    """
    intervals = validate_annotations(annotations, duration)
    rows = sorted(predictions, key=lambda row: row["t"])
    matrix = empty_matrix()
    ambiguous = 0.0
    for interval in intervals:
        start, end, truth = interval["start"], interval["end"], interval["target"]
        if truth == UNKNOWN:
            ambiguous += end - start
            continue
        accounted = 0.0
        for index, row in enumerate(rows):
            next_t = rows[index + 1]["t"] if index + 1 < len(rows) else duration
            left, right = max(start, row["t"]), min(end, next_t, row["t"] + max_gap)
            if right <= left:
                continue
            weight = right - left
            label = row["target"] if row["target"] in TARGETS else UNKNOWN
            matrix[truth][label] += weight
            accounted += weight
        matrix[truth][UNKNOWN] += max(0.0, end - start - accounted)
    result = summarize(matrix)
    result.update({"unit": "seconds", "ambiguous_seconds": ambiguous,
                   "annotated_seconds": sum(r["end"] - r["start"] for r in intervals),
                   "session_seconds": duration,
                   "annotation_coverage": sum(r["end"] - r["start"] for r in intervals) / duration if duration else 0})
    return result


def temporal_iou(first, second):
    intersection = max(0.0, min(first["end"], second["end"]) - max(first["start"], second["start"]))
    union = max(first["end"], second["end"]) - min(first["start"], second["start"])
    return intersection / union if union > 0 else 0.0


def evaluate_temporal_events(events, annotations, duration, thresholds=(0.1, 0.3, 0.5, 0.7)):
    """Avalia intervalos observáveis apenas nos trechos anotados pelo humano."""
    intervals = [row for row in validate_annotations(annotations, duration) if row["target"] != UNKNOWN]
    truth = [{"start": row["start"], "end": row["end"], "state": ATTENTION_BY_TARGET[row["target"]]}
             for row in intervals]
    predicted = []
    for event in events:
        if event.get("state") not in ATTENTION_BY_TARGET.values():
            continue
        for coverage in intervals:
            start, end = max(float(event["start"]), coverage["start"]), min(float(event["end"]), coverage["end"])
            if end > start:
                predicted.append({"start": start, "end": end, "state": event["state"]})

    states = tuple(ATTENTION_BY_TARGET.values())
    results = {}
    for threshold in thresholds:
        per_state = {}
        total_tp = total_fp = total_fn = 0
        for state in states:
            pred = [row for row in predicted if row["state"] == state]
            actual = [row for row in truth if row["state"] == state]
            candidates = sorted(((temporal_iou(p, a), pi, ai)
                                 for pi, p in enumerate(pred) for ai, a in enumerate(actual)), reverse=True)
            used_pred, used_actual = set(), set()
            for score, pi, ai in candidates:
                if score < threshold:
                    break
                if pi not in used_pred and ai not in used_actual:
                    used_pred.add(pi); used_actual.add(ai)
            tp, fp, fn = len(used_pred), len(pred) - len(used_pred), len(actual) - len(used_actual)
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = (2 * precision * recall / (precision + recall)
                  if precision is not None and recall is not None and precision + recall else 0.0)
            per_state[state] = {"tp": tp, "fp": fp, "fn": fn,
                                "precision": precision, "recall": recall, "f1": f1}
            total_tp += tp; total_fp += fp; total_fn += fn
        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else None
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else 0.0)
        results[str(threshold)] = {"tp": total_tp, "fp": total_fp, "fn": total_fn,
                                   "precision": precision, "recall": recall, "f1": f1,
                                   "per_state": per_state}
    best_ious = [max((temporal_iou(row, candidate) for candidate in predicted
                      if candidate["state"] == row["state"]), default=0.0) for row in truth]
    return {"thresholds": results, "mean_best_iou": sum(best_ious) / len(best_ious) if best_ious else None,
            "predicted_intervals": len(predicted), "reference_intervals": len(truth),
            "evaluated_seconds": sum(row["end"] - row["start"] for row in intervals),
            "note": "Avalia foco por alvo e dispersão externa apenas nos trechos anotados."}


def _joint_segments(first, second, duration):
    """Particiona somente o tempo coberto pelos dois avaliadores."""
    left = validate_annotations(first, duration)
    right = validate_annotations(second, duration)
    boundaries = sorted({value for row in (*left, *right)
                         for value in (row["start"], row["end"])})
    segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        middle = (start + end) / 2
        first_row = next((row for row in left if row["start"] <= middle < row["end"]), None)
        second_row = next((row for row in right if row["start"] <= middle < row["end"]), None)
        if first_row and second_row:
            segments.append({"start": start, "end": end,
                             "first": first_row["target"], "second": second_row["target"]})
    return segments, left, right


def _merge_segments(rows, keys):
    merged = []
    for row in rows:
        if (merged and abs(merged[-1]["end"] - row["start"]) < 1e-9
                and all(merged[-1].get(key) == row.get(key) for key in keys)):
            merged[-1]["end"] = row["end"]
            merged[-1]["duration"] = merged[-1]["end"] - merged[-1]["start"]
        else:
            value = dict(row)
            value["duration"] = value["end"] - value["start"]
            merged.append(value)
    return merged


def compare_annotators(first, second, duration, first_name="avaliador_1", second_name="avaliador_2"):
    """Concordancia temporal entre dois avaliadores independentes.

    As metricas principais usam apenas a cobertura conjunta determinada. Trechos
    com INDETERMINADO permanecem auditaveis, mas sao informados separadamente.
    """
    if first_name == second_name:
        raise ValueError("Selecione dois avaliadores diferentes")
    segments, first_rows, second_rows = _joint_segments(first, second, duration)
    labels = (*TARGETS, UNKNOWN)
    matrix = {label: {other: 0.0 for other in labels} for label in labels}
    disagreements, consensus = [], []
    joint_seconds = comparable_seconds = agreement_seconds = unknown_seconds = 0.0

    for row in segments:
        weight = row["end"] - row["start"]
        joint_seconds += weight
        matrix[row["first"]][row["second"]] += weight
        determined = row["first"] != UNKNOWN and row["second"] != UNKNOWN
        if determined:
            comparable_seconds += weight
            if row["first"] == row["second"]:
                agreement_seconds += weight
        else:
            unknown_seconds += weight
        if row["first"] != row["second"]:
            disagreements.append(dict(row))
        consensus.append({"start": row["start"], "end": row["end"],
                          "target": row["first"] if row["first"] == row["second"] else UNKNOWN,
                          "needs_review": row["first"] != row["second"],
                          "first": row["first"], "second": row["second"]})

    actual = {label: sum(matrix[label][other] for other in TARGETS) for label in TARGETS}
    predicted = {label: sum(matrix[other][label] for other in TARGETS) for label in TARGETS}
    expected = (sum(actual[label] * predicted[label] for label in TARGETS) / comparable_seconds ** 2
                if comparable_seconds else 0.0)
    observed = agreement_seconds / comparable_seconds if comparable_seconds else None
    kappa = ((observed - expected) / (1 - expected)
             if observed is not None and expected < 1 - 1e-12 else None)
    per_target_iou = {}
    for label in TARGETS:
        intersection = matrix[label][label]
        union = (sum(matrix[other][label] for other in TARGETS)
                 + sum(matrix[label][other] for other in TARGETS) - intersection)
        per_target_iou[label] = intersection / union if union else None

    return {
        "first": first_name, "second": second_name, "unit": "seconds",
        "matrix": matrix, "joint_seconds": joint_seconds,
        "joint_coverage": joint_seconds / duration if duration else 0.0,
        "comparable_seconds": comparable_seconds,
        "agreement_seconds": agreement_seconds,
        "agreement": observed, "kappa": kappa,
        "indeterminate_seconds": unknown_seconds,
        "first_annotated_seconds": sum(row["end"] - row["start"] for row in first_rows),
        "second_annotated_seconds": sum(row["end"] - row["start"] for row in second_rows),
        "per_target_iou": per_target_iou,
        "disagreements": _merge_segments(disagreements, ("first", "second")),
        "consensus_draft": _merge_segments(consensus, ("target", "needs_review", "first", "second")),
        "note": "Concordancia calculada apenas na cobertura conjunta; trechos indeterminados sao separados.",
    }


def evaluate_checks(rows):
    matrix = empty_matrix()
    for row in rows:
        matrix[row["expected"]][row["target"]] += 1
    result = summarize(matrix)
    result["unit"] = "samples"
    task_matrix = {truth: {pred: 0 for pred in ("DENTRO", "FORA", UNKNOWN)} for truth in ("DENTRO", "FORA")}
    for row in rows:
        truth = "FORA" if row["expected"] == "OUTRO_LOCAL" else "DENTRO"
        predicted = row.get("regions", {}).get("task", {}).get("target")
        if predicted is None:
            predicted = UNKNOWN if row["target"] == UNKNOWN else ("FORA" if row["target"] == "OUTRO_LOCAL" else "DENTRO")
        task_matrix[truth][predicted] += 1
    total = len(rows)
    task_correct = task_matrix["DENTRO"]["DENTRO"] + task_matrix["FORA"]["FORA"]
    known = total - task_matrix["DENTRO"][UNKNOWN] - task_matrix["FORA"][UNKNOWN]
    result["task_metrics"] = {"matrix": task_matrix, "total": total,
                               "accuracy": task_correct / total if total else None,
                               "coverage": known / total if total else None}
    return result
