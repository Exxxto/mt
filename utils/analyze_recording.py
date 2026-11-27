"""Read-only summary of a Stage 2 diagnostics session."""

import argparse
import json
from collections import Counter
from pathlib import Path


def _read_json(path):
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _read_events(path):
    events = []
    malformed = 0
    try:
        with path.open("rb") as stream:
            for raw_line in stream:
                if not raw_line.strip():
                    continue
                try:
                    event = json.loads(raw_line.decode("utf-8"))
                    if not isinstance(event, dict) or not isinstance(event.get("event_type"), str):
                        raise ValueError("not an event object")
                    events.append(event)
                except (UnicodeError, json.JSONDecodeError, ValueError):
                    malformed += 1
    except (OSError, UnicodeError):
        pass
    return events, malformed


def analyze_session(path, verbose=False):
    """Return a Russian text report, tolerating an incomplete recording."""
    folder = Path(path)
    metadata_path = folder / "metadata.json"
    events_path = folder / "events.jsonl"
    summary_path = folder / "summary.json"
    metadata = _read_json(metadata_path)
    summary = _read_json(summary_path)
    events, malformed = _read_events(events_path)
    missing = [name for name, file_path in (
        ("metadata.json", metadata_path), ("events.jsonl", events_path),
        ("summary.json", summary_path),
    ) if not file_path.is_file()]
    partial = bool(missing or malformed or metadata is None or summary is None
                   or (summary and summary.get("truncated")))

    types = Counter(event["event_type"] for event in events)
    phases = []
    reasons = Counter()
    refusals = Counter()
    purchase_reasons = Counter()
    purchase_confirmations = Counter()
    elixir_observations = []
    unknowns = 0
    repeated_frames = 0
    purchases = []
    sales = []
    upgrades = 0
    for event in events:
        kind = event["event_type"]
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if kind in ("phase", "phase_unknown"):
            phase = data.get("phase", event.get("phase"))
            phases.append(str(phase) if phase is not None else "unknown")
        reason = data.get("reason")
        if kind == "decision" and reason:
            reasons[str(reason)] += 1
        if kind == "purchase_refused" and reason:
            refusals[str(reason)] += 1
        if kind == "buy_attempt" and data.get("selection_reason"):
            purchase_reasons[str(data["selection_reason"])] += 1
        if kind in ("decision", "state_snapshot", "purchase_refused"):
            elixir = data.get("elixir")
            if isinstance(elixir, dict):
                value, certainty = elixir.get("value"), elixir.get("certainty", "unknown")
                elixir_observations.append((event.get("elapsed_ms"), value, certainty))
                if value is None or certainty == "unknown":
                    unknowns += 1
        if kind == "phase_unknown":
            unknowns += 1
        if kind == "frame_reused":
            repeated_frames += 1
        if kind in ("buy_outcome", "buy_simulated"):
            purchases.append((kind, data))
            if data.get("confirmation"):
                purchase_confirmations[str(data["confirmation"])] += 1
        if kind.startswith("sale_"):
            sales.append((kind, data))
        if kind == "calculated_upgrade" and data.get("threshold_reached"):
            upgrades += 1

    session_id = (metadata or {}).get("session_id", folder.name)
    lines = ["Диагностика игровой сессии", f"Сессия: {session_id}"]
    if metadata:
        lines.append(f"Начало: {metadata.get('started_at', 'неизвестно')} | стратегия: {metadata.get('strategy', 'неизвестна')}")
    if summary:
        counters = summary.get("counters") if isinstance(summary.get("counters"), dict) else {}
        lines.append(
            f"Итог: {summary.get('finish_reason', 'неизвестен')} | длительность: "
            f"{summary.get('duration_ms', 'неизвестна')} мс | событий: "
            f"{counters.get('events_written', len(events))} | усечена: "
            f"{'да' if summary.get('truncated') else 'нет'}"
        )
    else:
        lines.append("Итог: summary.json отсутствует или повреждён")
    if missing:
        lines.append("Отсутствует: " + ", ".join(missing))
    lines.append(f"Состояние: {'неполная' if partial else 'полная'} запись | повреждённых строк: {malformed}")
    lines.append("Фазы: " + (" → ".join(phases) if phases else "нет наблюдений"))
    lines.append("Решения: " + (", ".join(f"{key}={value}" for key, value in sorted(reasons.items())) or "нет")
                 + f" (всего {types['decision']})")
    if refusals:
        lines.append("Отказы покупки: " + ", ".join(f"{key}={value}" for key, value in sorted(refusals.items())))
    if elixir_observations:
        elixir_history = " → ".join(f"{value!r} ({certainty})" for _, value, certainty in elixir_observations)
        lines.append(f"Эликсир: {elixir_history}")
    else:
        lines.append("Эликсир: нет наблюдений")
    if purchase_reasons:
        lines.append("Причины попыток покупки: " + ", ".join(
            f"{key}={value}" for key, value in sorted(purchase_reasons.items())))
    if purchase_confirmations:
        lines.append("Уверенность исходов покупки: " + ", ".join(
            f"{key}={value}" for key, value in sorted(purchase_confirmations.items())))
    lines.append(f"Покупки: {len(purchases)} событий исхода; повышения порога (расчёт): {upgrades}")
    lines.append(f"Продажа: {len(sales)} связанных событий; повторно использованных кадров: {repeated_frames}; неизвестных значений: {unknowns}")
    errors = Counter(event["event_type"] for event in events
                     if event["event_type"] in ("adb_error", "screenshot_error"))
    lines.append("Ошибки: " + (", ".join(f"{key}={value}" for key, value in sorted(errors.items()))
                                or "не зарегистрированы"))

    if verbose:
        lines.append("Хронология:")
        if not events:
            lines.append("  нет корректных событий")
        for event in events:
            lines.append(
                f"  +{event.get('elapsed_ms', '?')} мс #{event.get('event_id', '?')} "
                f"{event['event_type']} frame={event.get('frame_id')} "
                f"action={event.get('action_id')} data="
                f"{json.dumps(event.get('data', {}), ensure_ascii=False, sort_keys=True)}"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Read a Merge Tactics diagnostics recording")
    parser.add_argument("session_dir", help="path to a recording session directory")
    parser.add_argument("--verbose", action="store_true", help="include the event timeline")
    args = parser.parse_args()
    print(analyze_session(args.session_dir, verbose=args.verbose))


if __name__ == "__main__":
    main()
