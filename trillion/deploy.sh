#!/usr/bin/env bash
# deploy.sh: start one rack's SGLang server on every one of its 18 trays, as a prefill or a decode rack.
# run on each tray with RANK=0..17 and HEAD set to the first tray, for example
#   deploy.sh prefill 8 1.02      deploy.sh decode 2 10.66      deploy.sh decode 4 1.87      deploy.sh decode 8 0.33
ROLE=$1 TP=$2 RATIO=$3
PORT=30000
EXTRA="--chunked-prefill-size 16384 --max-prefill-tokens 16384 --enable-hierarchical-cache --disaggregation-bootstrap-port 8998"
[ "$ROLE" = decode ] && PORT=30100 && EXTRA="--dcp-size $TP"
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 sglang serve \
  --model-path moonshotai/Kimi-K3 --revision f831ab66814297da540d832a5235f8e904f29d06 --trust-remote-code \
  --nnodes 18 --node-rank "$RANK" --dist-init-addr "$HEAD:20000" \
  --tp-size 72 --attn-dp-size $((72 / TP)) --ep-size 72 --moe-a2a-backend megamoe \
  --enable-eplb --ep-num-redundant-experts 112 --mem-fraction-static 0.85 \
  --kv-cache-dtype fp8_e4m3 --mamba-full-memory-ratio "$RATIO" \
  --disaggregation-mode "$ROLE" --disaggregation-transfer-backend nixl \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-metrics \
  --host 0.0.0.0 --port "$PORT" $EXTRA
