"""capacity.py: what Kimi K3 costs to serve on GB300 NVL72 racks, worked out from config.py alone."""
import math

import config as cfg

KDA_LAYERS, MLA_LAYERS = cfg.LAYERS.count("kda"), cfg.LAYERS.count("mla")


def request_bytes(tokens):
    """(KDA state, MLA latent) bytes one request keeps in the pool at a context length."""
    state = KDA_LAYERS * (cfg.KDA_STATE * cfg.STATE_BYTES + cfg.KDA_CONV * cfg.CONV_BYTES)
    return state, MLA_LAYERS * cfg.LATENT * cfg.KV_BYTES * tokens


SLOT_BYTES = cfg.SLOTS * cfg.MOE_LAYERS * cfg.EXPERT_BYTES          # the routed experts one GPU holds
ROUTED_FLOPS = 2 * cfg.TOP_K * cfg.EXPERT_PARAMS * cfg.MOE_LAYERS   # per token, inside the experts


def weights():
    """Bytes of weights on each GPU, for three ways of placing Kimi K3 on a rack."""
    experts = cfg.EXPERTS * cfg.MOE_LAYERS * cfg.EXPERT_BYTES
    return {"whole model on every 8 GPUs": (cfg.TRUNK + experts) / 8,
            "experts over 72, trunk on each": SLOT_BYTES + cfg.TRUNK,   # attention fully data parallel
            "experts over 72, trunk over 8": SLOT_BYTES + cfg.TRUNK / 8}


def token_seconds(attention):
    """Seconds of peak compute for one token."""
    return ROUTED_FLOPS / cfg.FP8 + (cfg.ACTIVE_FLOPS - ROUTED_FLOPS + attention) / cfg.BF16


def wire_bytes(tp):
    """Bytes one GPU sends per request each decode step, in a group of tp GPUs."""
    reduce = 2 * (tp - 1) / tp * cfg.HIDDEN * 2 * len(cfg.LAYERS)   # one bf16 all-reduce per layer
    merge = (tp - 1) / tp * cfg.HEADS * (cfg.LATENT + cfg.KV_RANK) * 2 * MLA_LAYERS   # DCP partial outputs
    reached = cfg.GPUS * (1 - (1 - 1 / cfg.GPUS) ** cfg.TOP_K)     # GPUs one token's 16 experts sit on
    return reduce + merge + reached * cfg.MOE_WIDTH * 3 * cfg.MOE_LAYERS / tp   # FP8 out, bf16 back


def decode_step(tokens, tp, batch):
    """(HBM, FLOP, wire) seconds of one decode step for batch requests on a group of tp GPUs."""
    state, latent = request_bytes(tokens)
    moved = cfg.TRUNK / tp + SLOT_BYTES + batch * (2 * state + latent) / tp   # state is read, then written
    attention = 2 * cfg.HEADS * (cfg.LATENT + cfg.KV_RANK) * MLA_LAYERS * tokens   # scores, then values
    return moved / cfg.HBM_BW, batch / tp * token_seconds(attention), batch * wire_bytes(tp) / cfg.NVLINK


def per_rack(tokens, tp):
    """Requests one rack holds with groups of tp GPUs."""
    state, latent = request_bytes(tokens)
    pool = cfg.MEM_FRACTION * cfg.HBM - cfg.TRUNK / tp - SLOT_BYTES     # the trunk splits, the slots do not
    memory = cfg.POOL_FILL * cfg.GPUS * pool / (state + latent)        # TP splits the state, DCP the latent
    hbm0, _, _ = decode_step(tokens, tp, 0)                            # the weights alone
    hbm1, flop1, wire1 = decode_step(tokens, tp, 1)                    # the cost of one request
    batch = min((cfg.STEP_BUDGET - hbm0) / (hbm1 - hbm0 + wire1), cfg.STEP_BUDGET / (flop1 + wire1))
    return int(min(memory, cfg.GPUS / tp * batch))


def best_layout(tokens):
    """(tp, requests per rack) for the best group size."""
    return max(((tp, per_rack(tokens, tp)) for tp in (1, 2, 4, 8)), key=lambda pair: pair[1])


def prefill_seconds(context):
    """GPU-seconds to prefill one new token that attends over context earlier tokens."""
    attention = 2 * cfg.HEADS * (2 * cfg.HEAD_DIM + cfg.ROPE) * MLA_LAYERS * context   # expanded heads
    return token_seconds(attention) / cfg.MFU


def bill(name):
    """Print what one request of a class costs: bytes held, bytes moved, prefill work and handoff time."""
    _, prompt, cached, out, _ = cfg.CLASSES[name]
    (state, latent), new = request_bytes(prompt + out), prompt - cached
    print(f"one {name} request: {prompt:,} prompt tokens, {cached:,} cached, {out:,} out")
    for label, value in (("KDA state, fixed size", f"{state / 1e6:,.1f} MB"),
                         (f"MLA latent at {prompt + out:,} tokens", f"{latent / 1e6:,.1f} MB"),
                         ("read and written every decode step", f"{(2 * state + latent) / 1e6:,.1f} MB"),
                         (f"prefill of {new:,} new tokens", f"{new * prefill_seconds(cached + new / 2):.3f} GPU-s"),
                         ("handoff to decode over one NIC", f"{sum(request_bytes(prompt)) / cfg.NIC * 1e3:.1f} ms")):
        print(f"  {label:<40}{value:>12}")


def size_cluster(users=100_000):
    """{class: (tp, decode racks, step seconds, arrivals per second, prefill GPUs)} by Little's law."""
    plan = {}
    for name, (share, prompt, cached, out, ttft) in cfg.CLASSES.items():
        tp, rack = best_layout(prompt + out)                     # the pool holds the finished answer
        hbm, flops, wire = decode_step(prompt + out, tp, rack * tp / cfg.GPUS)
        step = max(hbm, flops) + wire                            # the all-reduces sit between layers
        arrivals = users * share / (ttft + out * step)           # Little's law, L = lambda W, for lambda
        new = prompt - cached
        plan[name] = (tp, users * share / rack, step, arrivals, arrivals * new * prefill_seconds(cached + new / 2))
    return plan


def pools(plan):
    """{pool: (whole racks, state to latent ratio)}: the decode pools by layout, and prefill over every class."""
    need, state, latent = {}, {}, {}
    for name, (tp, racks, _, _, gpus) in plan.items():
        share, prompt, _, out, _ = cfg.CLASSES[name]
        for key, tokens, n in ((f"decode tp{tp}", prompt + out, racks), ("prefill", prompt, gpus / cfg.GPUS / cfg.PREFILL_BUSY)):
            s, l = request_bytes(tokens)
            need[key], state[key], latent[key] = need.get(key, 0) + n, state.get(key, 0) + share * s, latent.get(key, 0) + share * l
    return {key: (math.ceil(need[key]), round(state[key] / latent[key], 2)) for key in sorted(need)}
