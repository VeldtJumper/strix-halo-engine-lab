"""Short Markdown renderings of receipts (the JSON receipt is the record of authority)."""


def _f(x, digits=5):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    return f"{x:+.{digits}f}" if isinstance(x, float) else str(x)


def _statement(text):
    return f" Reading: {text}." if text else ""


def _table(endpoints, guards):
    lines = ["| endpoint | estimate | upper bound | margin | pass |", "|---|---:|---:|---:|---|"]
    for e, v in endpoints.items():
        lines.append(f"| {e} {v['label']} | {_f(v['estimate'])} | {_f(v['ucb'])} | {_f(v['margin'])} | "
                     f"{'yes' if v['passed'] else 'NO'} |")
    for g, v in guards.items():
        lines.append(f"| {g} stratum points of {v['endpoint']} < {_f(v['limit'], 4)} | "
                     + ", ".join(f"{s} {_f(p, 4)}" for s, p in v["stratum_points"].items())
                     + f" | | | {'yes' if v['passed'] else 'NO'} |")
    return lines


def render_md(r):
    rep = r["report"]
    receipts = r.get("arm_model_receipt_sha256", {})
    lines = [f"# {r['option']}: {r['verdict']}", "",
             f"{'Qualification' if r['qualification'] else 'SEARCH split: not a qualification'}; "
             f"reference `{r['reference']}`, control `{r['control']}`"
             + (f", composition of {', '.join(r['members'])}" if r.get("members") else "") + ".",
             f"Lengths {r['lengths']}; {r['parents_used']} of {r['parents_in_scope']} parents used "
             f"({r['parents_used_per_stratum']}), "
             f"{rep['positions'] if rep else 0} positions; one-sided level {r['level']:.5f} (alpha {r['alpha']}, k {r['k']}).",
             f"Plan `{r['plan_sha256'][:16]}` (frozen {r['plan_frozen_utc']}), manifest `{r['manifest_sha256'][:16]}`, "
             f"binary {', '.join(b[:16] for b in r['binary_sha256'])}.",
             *(["Model receipt sha256: " + "; ".join(
                 f"`{name}` `{receipts.get(name, '-')}`" for name in dict.fromkeys(
                     n for n in (r["reference"], r["control"], r["option"]) if n)) + "."]
               if "arm_model_receipt_sha256" in r else []), "",
             f"Deciding margins: {r.get('tier', 'exact-class')}." + _statement(r.get("statement"))
             + (" Every endpoint is the mean of the stratum means (strata weigh equally)."
                if r.get("stratum_weights") == "equal" else ""), ""]
    lines += _table(r["endpoints"], r.get("guards", {}))
    sec = r.get("secondary")
    if sec:
        lines += ["", f"Also reported (does not decide the verdict): {sec['tier']} margins -> {sec['verdict']}."
                  + _statement(sec.get("statement")), ""]
        lines += _table(sec["endpoints"], sec["guards"])
    if rep:
        m = rep["mean_parent"]
        lines += ["", f"Mean per parent: dNLL {m['dnll']:+.5f} nats/token ({rep['ppl_change_percent']:+.3f} % perplexity), "
                  f"overlap loss {m['loss']:.5f}, KL(ref||opt) {m['kl_rc']:.2e}, KL(opt||ref) {m['kl_cr']:.2e}, "
                  f"top-1 disagreement {m['disagree']:.4f}.",
                  f"Positions: overlap mean {rep['overlap_position_mean']:.5f}, 1 % quantile {rep['overlap_q01']:.4f}, "
                  f"0.1 % quantile {rep['overlap_q001']:.4f}; max KL {rep['kl_rc_max']:.3e}; near ties "
                  f"{rep['near_tie_fraction']:.3%}.",
                  f"Worst parent {rep['worst_parent']['parent']} {rep['worst_parent']['dnll']:+.4f}, "
                  f"best {rep['best_parent']['parent']} {rep['best_parent']['dnll']:+.4f}."]
    if r["notes"]:
        lines += ["", "Problems:"] + [f"- {n}" for n in r["notes"][:12]]
        if len(r["notes"]) > 12:
            lines.append(f"- ... {len(r['notes']) - 12} more in the JSON receipt")
    return "\n".join(lines) + "\n"


def render_summary_md(s, results):
    lines = [f"# Qualification summary ({s['split']})", "",
             f"Plan `{s['plan_sha256'][:16]}`, manifest `{s['manifest_sha256'][:16]}`, k = {s['k']}, "
             f"parents in scope {s['parents_in_scope']}.", "",
             "| option | verdict | " + " | ".join(next(iter(results.values()))["endpoints"]) + " |" if results else "",
             "|---|---|" + "---:|" * len(next(iter(results.values()))["endpoints"]) if results else ""]
    for row in s["verdicts"]:
        r = results[row["option"]]
        cells = [f"{_f(v['estimate'])} (ub {_f(v['ucb'])})" for v in r["endpoints"].values()]
        lines.append(f"| {row['option']} | {row['verdict']} | " + " | ".join(cells) + " |")
    if s.get("control_means"):
        lines += ["", "Control means (input of plan calibrate): "
                  + ", ".join(f"{k} {v:.6g}" for k, v in s["control_means"].items())]
    if s["global_problems"]:
        lines += ["", "Global problems:"] + [f"- {p}" for p in s["global_problems"][:12]]
    return "\n".join(lines) + "\n"
