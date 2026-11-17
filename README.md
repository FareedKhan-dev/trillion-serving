# trillion-serving

The code behind my post on serving Kimi K3, a 2.78T-parameter model, to 100,000 parallel streams on GB300 NVL72 racks.

[Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) keeps a fixed recurrent state in 69 of its layers and a growing latent cache in the other 24 ([tech report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)). The post starts from what one request costs in bytes and builds up to a capacity planner, a fleet router and a load test on 21 racks. This repository holds that code, the numbers behind every figure and the post's numpy cells.

## Layout

```
trillion-serving/
  trillion/
    config.py        every number the planner reads: the model, the rack, my serving choices, the request mix
    capacity.py      the planner: bytes a request keeps, weights per GPU, one decode step, racks for the mix
    router.py        the fleet router: a hash ring over the prefill racks, class budgets, admission on pool fill
    loadgen.py       holds streams open against the router and times every token
    deploy.sh        starts one rack's SGLang server on each of its trays, as a prefill or a decode rack
  results/
    capacity.json    the planner's numbers
    run.json         the load test's primitives, the inputs to derive.py
    loadtest.json    every load test number in the post, from run.json and the planner
    facts.json       the facts the post quotes, each with its source
    derive.py        rebuilds loadtest.json from run.json and the planner
  notebook.ipynb     the post's numpy and planner cells, in order
```

## Setup

```bash
git clone https://github.com/FareedKhan-dev/trillion-serving
cd trillion-serving
uv sync
```

`pyproject.toml` holds the post's `uv add` line: `sglang[all]==0.5.23`, `aiohttp` and `huggingface_hub`. Every tray reads the weights from shared storage, at the revision the post pins:

```bash
hf download moonshotai/Kimi-K3 --revision f831ab66814297da540d832a5235f8e904f29d06 --local-dir /models/kimi-k3
```

## The planner

The planner needs no GPU. The notebook runs the post's cells in order:

```bash
uv run --with jupyterlab jupyter lab
```

It also runs on its own. Start `uv run python` in the `trillion` folder and call it:

```python
from capacity import bill, pools, size_cluster
bill("chat")                    # what one chat request costs, in bytes, time and GPU-seconds
print(pools(size_cluster()))    # whole racks and the pool split for 100,000 streams
```

## The load test

What the code expects:

- 21 GB300 NVL72 racks with the model at the pinned revision on every tray.
- 13 prefill racks reachable as `prefill-0` to `prefill-12` on port 30000, and 8 decode racks as `decode-0` to `decode-7` on port 30100.
- The pool split each rack starts with, from the planner: 1.02 for prefill, 10.66 for the three chat racks on TP2, 1.87 for the document rack on TP4 and 0.33 for the four agent, long and max racks on TP8.

`deploy.sh` reads two environment variables, `RANK` (the tray, 0 to 17) and `HEAD` (the first tray's address). The router and the load generator read none.

1. On every tray of every rack, start its server, for example a chat rack:

   ```bash
   RANK=0 HEAD=10.0.0.1 ./trillion/deploy.sh decode 2 10.66
   ```

2. Start 24 router processes, since each one holds a 24th of every class budget, and spread the clients over them:

   ```bash
   cd trillion
   uv run python -m aiohttp.web -H 0.0.0.0 -P 8000 router:app
   ```

3. Run the load generator from 16 hosts with 8 processes each. A process takes the router URL, its share of the 100,000 streams, its first stream id and the seconds of the measured window after the 10 minute ramp, and writes its TTFTs and gap histogram to `loadgen-<first>.json`:

   ```bash
   uv run python trillion/loadgen.py http://router:8000/v1/chat/completions 782 0 1800
   ```

4. Put the run's measurements into `results/run.json`, with the same keys, and rebuild every number:

   ```bash
   uv run results/derive.py
   ```

## Not done yet

- The load test values in `results/` stay provisional until a full run on 21 racks replaces them.
- Nothing here spreads clients over the 24 router processes, so that front is up to the cluster.
- `deploy.sh` reserves no memory for the all-to-all backend's buffers, which SGLang's cookbook warns take back the KV memory that DCP frees.
- Speculative decoding with Kimi K3's DSpark draft model is off.
- Only text is served, not the vision encoder.

The weights come under the Kimi K3 License.
