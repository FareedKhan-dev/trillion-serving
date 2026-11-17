"""config.py: every number the planner reads, from the model, the rack and our own serving choices."""
MODEL, REVISION = "moonshotai/Kimi-K3", "f831ab66814297da540d832a5235f8e904f29d06"
PARAMS, CHECKPOINT = 2_779_931_837_184, 1_560_936_091_448   # parameters, and bytes on disk
LAYERS = ["kda", "kda", "kda", "mla"] * 23 + ["mla"]        # 93 layers: 69 KDA, 24 MLA
HIDDEN, HEADS, HEAD_DIM = 7168, 96, 128
KV_RANK, ROPE = 512, 64                 # MLA caches a 512 latent and a 64 wide shared key
LATENT = KV_RANK + ROPE
KDA_STATE = HEADS * HEAD_DIM * HEAD_DIM # values a KDA layer keeps per request
KDA_CONV = 3 * HEADS * HEAD_DIM * 3     # short convolution: q, k, v, last 3 steps
EXPERTS, TOP_K, MOE_LAYERS = 896, 16, 92                    # layer 0 is the dense one
MOE_WIDTH, EXPERT_PARAMS = 3584, 3 * 3584 * 3072            # gate, up, down in the latent
EXPERT_BYTES = EXPERT_PARAMS * 4.25 / 8                     # MXFP4: 4 bits plus a scale per 32
TRUNK = 113.49e9                        # every always-active weight, bf16 bytes
ACTIVE_FLOPS = 2 * 104.2e9              # FLOPs per token, two per active parameter

# One GB300 NVL72 rack, per GPU unless the name says otherwise
GPUS, HBM, HBM_BW = 72, 276.6 * 2**30, 8e12   # GPUs, usable bytes, bytes a second
NVLINK, NIC = 900e9, 100e9              # bytes a second each way, NVLink and one 800 Gb/s port
FP8, BF16 = 5e15, 2.5e15                # dense FLOPs a second
RACK_KW = 140                           # rated power of one rack

# Our serving choices
MEM_FRACTION, POOL_FILL = 0.85, 0.9     # HBM for weights and pool, and the fill we plan to
PREFILL_BUSY = 0.75                     # the most of a prefill rack we plan to use
STEP_BUDGET = 0.05                      # one decode step: 20 tokens a second per stream
SLOTS = 14                              # expert slots per GPU per layer, 1,008 a rack
KV_BYTES, STATE_BYTES, CONV_BYTES = 1, 4, 2   # FP8 latent, fp32 state, bf16 convolution
MFU = 0.5                               # share of peak FLOPs a prefill GPU reaches
CLASSES = {                             # share of streams, prompt, cached prefix, output, TTFT goal (s)
    "chat": (0.60, 2_048, 1_536, 1_000, 1),
    "agent": (0.25, 49_152, 45_056, 1_500, 2),
    "doc": (0.10, 16_384, 12_288, 1_000, 4),
    "long": (0.045, 262_144, 253_952, 2_000, 20),
    "max": (0.005, 1_000_000, 983_616, 2_000, 60),
}
