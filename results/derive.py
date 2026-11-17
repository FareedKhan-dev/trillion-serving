"""derive.py: every number of the load test the post prints, derived from the authored primitives (results/run.json)
and the planner (trillion/capacity.py), so a real run that replaces run.json moves every number at once.

What the planner decides, it decides here too: each decode pool's real step comes from capacity.decode_step at the
batch the pool actually carried, inter-token times are that step times the authored overhead, arrivals follow from
Little's law, and the prefill load follows from the arrivals and the cold sessions. A derived leaf stays provisional.

Writes results/loadtest.json.

    uv run results/derive.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "trillion"))
import capacity as cap  # noqa: E402
import config as cfg  # noqa: E402

SESSION_TURNS = 256                                   # loadgen.py starts a new session every 256 turns


def plain(x):
    if isinstance(x, dict):
        if "v" in x and "provisional" in x:
            return x["v"]
        return {k: plain(v) for k, v in x.items()}
    return x


def pool_steps(users=100_000):
    """{tp: (hbm, flops, wire, step) seconds} for each decode pool at the batch it carried in the run."""
    plan = cap.size_cluster(users)
    racks = {k: r for k, (r, _) in cap.pools(plan).items()}
    out = {}
    for tp in sorted({p[0] for p in plan.values()}):
        groups = racks[f"decode tp{tp}"] * cfg.GPUS / tp
        members = [(n, users * cfg.CLASSES[n][0] / groups) for n in plan if plan[n][0] == tp]
        hbm = flops = wire = 0.0
        for k, (name, batch) in enumerate(members):
            _, prompt, _, out_tokens, _ = cfg.CLASSES[name]
            h, f, w = cap.decode_step(prompt + out_tokens, tp, batch)
            base = cap.decode_step(prompt + out_tokens, tp, 0)[0]   # the weights, counted once per step
            hbm += h - (base if k else 0)
            flops, wire = flops + f, wire + w
        out[tp] = (hbm, flops, wire, max(hbm, flops) + wire)
    return out, plan


def main():
    run = plain(json.loads((ROOT / "results" / "run.json").read_text(encoding="utf-8")))
    steps, plan = pool_steps()
    window = run["window_s"]
    classes, totals = {}, {"arrivals": 0.0, "out": 0.0, "prompt": 0.0, "uncached": 0.0, "prefill_gpus": 0.0}
    for name, (share, prompt, cached, out, goal) in cfg.CLASSES.items():
        tp = plan[name][0]
        step = steps[tp][3]
        itl = {k: step * f * 1e3 for k, f in run["itl_over_step"].items()}             # ms
        ttft = run["ttft"][name]
        streams = 100_000 * share * run["in_flight"]["mean"] / 100_000
        arrivals = streams / (ttft["mean"] + out * itl["mean"] / 1e3)               # Little's law
        cold = 1 / SESSION_TURNS                                                     # the first turn of a session
        uncached = (1 - cold) * (prompt - cached) + cold * prompt
        hit = 1 - uncached / prompt
        gpus = arrivals * ((1 - cold) * (prompt - cached) * cap.prefill_seconds(cached + (prompt - cached) / 2)
                           + cold * prompt * cap.prefill_seconds(prompt / 2))
        classes[name] = {"tp": tp, "streams": streams, "itl": itl, "ttft": ttft, "arrivals": arrivals,
                         "requests": arrivals * window, "out_tok_s": arrivals * out, "hit": hit,
                         "goodput": run["goodput"][name], "ttft_goal": goal, "step_ms": step * 1e3,
                         "floor_ms": max(steps[tp][0], steps[tp][1]) * 1e3}
        totals["arrivals"] += arrivals
        totals["out"] += arrivals * out
        totals["prompt"] += arrivals * prompt
        totals["uncached"] += arrivals * uncached
        totals["prefill_gpus"] += gpus
    capj = plain(json.loads((ROOT / "results" / "capacity.json").read_text(encoding="utf-8")))
    cl = capj["cluster"]
    racks = cl["racks"]
    kw = cl["decode_racks"] * run["rack_kw"]["decode"] + cl["prefill_racks"] * run["rack_kw"]["prefill"]
    requests = sum(c["requests"] for c in classes.values())
    good = sum(c["requests"] * c["goodput"] for c in classes.values()) / requests
    joules = kw * 1e3 / totals["out"]
    # pool fill on each decode pool: memory used over memory the pool has, at the run's streams
    fills = {}
    for tp, (r, _) in ((int(k.split("tp")[1]), v) for k, v in cap.pools(plan).items() if k.startswith("decode")):
        room = r * cfg.GPUS * (cfg.MEM_FRACTION * cfg.HBM - cfg.TRUNK / tp - cap.SLOT_BYTES)
        used = sum(100_000 * cfg.CLASSES[n][0] * sum(cap.request_bytes(cfg.CLASSES[n][1] + cfg.CLASSES[n][3]))
                   for n in plan if plan[n][0] == tp)
        fills[f"tp{tp}"] = used / room
    # the baseline: SGLang's published recipe, 8-GPU instances holding the whole model, one pool split for all
    w = capj["weights"]
    pool_gpu = (cfg.MEM_FRACTION * cfg.HBM) - w["copy8_gb"] * 1e9
    ratio = run["baseline"]["ratio"]
    state_gpu = pool_gpu * ratio / (1 + ratio)
    state_req_gpu = cap.request_bytes(0)[0] / 8
    slots = int(state_gpu / state_req_gpu)
    ssm = slots * cfg.HEADS / 8 * cfg.HEAD_DIM ** 2 * cfg.STATE_BYTES * cap.KDA_LAYERS
    conv = slots * 3 * 3 * cfg.HEADS / 8 * cfg.HEAD_DIM * cfg.CONV_BYTES * cap.KDA_LAYERS
    kv_gpu = pool_gpu - ssm - conv
    tokens = int(kv_gpu * 8 / (cap.MLA_LAYERS * cfg.LATENT * cfg.KV_BYTES))       # the KV half of one instance
    # a running request holds its prompt and, on average, half its output; the router mixes every class
    ctx = {n: c[1] + c[3] / 2 for n, c in cfg.CLASSES.items()}
    mean_ctx = sum(cfg.CLASSES[n][0] * ctx[n] for n in ctx)
    held = min(slots, int(tokens / mean_ctx))                                      # the half that fills first
    # the same split serving one class alone, the contrast behind H1: chats run out of state, agents of latent
    alone = {}
    for n in ("chat", "agent"):
        k = min(slots, int(tokens / ctx[n]))
        alone[n] = {"held": k, "state_used": k / slots, "latent_used": k * ctx[n] / tokens}
    base = dict(run["baseline"], slots=slots, ssm_gib=ssm / 2**30, conv_gib=conv / 2**30, kv_gib=kv_gpu / 2**30,
                tokens=tokens, mean_ctx=mean_ctx, per_instance=held, mamba_usage=held / slots,
                token_usage=held * mean_ctx / tokens, running=held * run["baseline"]["instances_decode"],
                pool_gib=pool_gpu / 2**30, alone=alone,
                per_expert=held * cfg.TOP_K / cfg.EXPERTS)                     # tokens one expert sees a step
    base["waiting"] = 100_000 - base["running"]
    base["queue_per_instance"] = round(base["waiting"] / base["instances_decode"])
    # a down prefill rack moves every session of its ring share that sends a turn during the outage
    share = capj["teach"]["ring"]["share"][f"http://{run['failover']['rack']}:30000"]   # the router's own hashing
    stay = {n: c["ttft"]["mean"] + cfg.CLASSES[n][3] * c["itl"]["mean"] / 1e3 for n, c in classes.items()}
    moved = sum(c["streams"] * share * min(1, run["failover"]["down_s"] / stay[n]) for n, c in classes.items())
    failover = dict(run["failover"], sessions=round(moved))
    # the once-a-minute samples behind the load plot must agree with the summary lines the post prints
    tl, ramp_min = run["timeline"], run["ramp_s"] // 60
    assert len(tl["ramp"]) == ramp_min and len(tl["window"]) == len(tl["chat_p99"]) == window // 60
    assert (min(tl["window"]), round(sum(tl["window"]) / len(tl["window"])), max(tl["window"])) == \
        (run["in_flight"]["min"], run["in_flight"]["mean"], run["in_flight"]["max"])
    assert tl["chat_p99"][run["burst"]["at_min"] - ramp_min] == max(tl["chat_p99"]) == run["burst"]["chat_p99_during"]
    assert tl["window"][run["failover"]["at_min"] - ramp_min] == run["in_flight"]["min"]
    timeline = {"in_flight": tl["ramp"] + tl["window"], "chat_p99": tl["chat_p99"], "window_from_min": ramp_min}
    out = {
        "window_min": window / 60, "ramp_min": run["ramp_s"] / 60, "in_flight": run["in_flight"],
        "classes": classes, "pool_steps_ms": {f"tp{k}": {"hbm": v[0] * 1e3, "flops": v[1] * 1e3, "wire": v[2] * 1e3,
                                                         "step": v[3] * 1e3} for k, v in steps.items()},
        "requests": requests, "arrivals": totals["arrivals"], "output_tok_s": totals["out"], "output_m": totals["out"] / 1e6,
        "input_tok_s": totals["prompt"], "uncached_tok_s": totals["uncached"],
        "hit_by_tokens": 1 - totals["uncached"] / totals["prompt"], "uncached_per_output": totals["uncached"] / totals["out"],
        "prefill_busy": totals["prefill_gpus"] / (cl["prefill_racks"] * cfg.GPUS),
        "prefill_gap_pts": (totals["prefill_gpus"] / (cl["prefill_racks"] * cfg.GPUS) - cfg.PREFILL_BUSY) * 100,
        "decode_tok_s_gpu": totals["out"] / (cl["decode_racks"] * cfg.GPUS), "goodput": good,
        "goodput_gap_ms": run["goodput_gap_ms"],
        "cluster_kw": kw, "joules_per_token": joules, "kwh_per_m": joules * 1e6 / 3.6e6,
        "gpu_s_per_1k": racks * cfg.GPUS / totals["out"] * 1e3, "fills": fills, "baseline": base,
        "hit_tiers": {"hbm": run["hit_from_hbm"], "lpddr": 1 - run["hit_from_hbm"]},
        "failover": failover, "secondary_share": run["secondary_share"],
        "burst": dict(run["burst"], chat_p99_delta_ms=(run["burst"]["chat_p99_during"] - run["ttft"]["chat"]["p99"]) * 1e3),
        "eplb": run["eplb"], "errors": run["errors"], "ablation": run["ablation"], "started": run["started"],
        "router_replicas": run["router_replicas"], "hosts": run["hosts"], "procs_per_host": run["procs_per_host"],
        "timeline": timeline,
    }

    def wrap(x):
        return {k: wrap(v) for k, v in x.items()} if isinstance(x, dict) else {"v": x, "provisional": True}
    (ROOT / "results" / "loadtest.json").write_text(json.dumps(wrap(out), indent=1) + "\n", encoding="utf-8", newline="\n")
    for k in ("requests", "arrivals", "output_tok_s", "hit_by_tokens", "uncached_per_output", "prefill_busy",
              "decode_tok_s_gpu", "goodput", "cluster_kw", "kwh_per_m", "gpu_s_per_1k"):
        print(f"{k:<22}{out[k]:,.4f}")
    for n, c in classes.items():
        print(f"{n:<6} tp{c['tp']} step {c['step_ms']:.1f} floor {c['floor_ms']:.1f} itl p50 {c['itl']['p50']:.1f} "
              f"p99 {c['itl']['p99']:.1f} arrivals {c['arrivals']:.1f} hit {c['hit']:.3f}")
    print("fills", {k: round(v, 3) for k, v in fills.items()})
    print("baseline", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in base.items()})


if __name__ == "__main__":
    main()
