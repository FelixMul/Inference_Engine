# Model architecture constants derived from config.json

HIDDEN_SIZE = 2048
NUM_LAYERS = 40
VOCAB_SIZE = 248320
RMS_NORM_EPS = 1e-6

# Layers with full (standard) attention; rest use linear attention
FULL_ATTN_LAYERS = frozenset([3, 7, 11, 15, 19, 23, 27, 31, 35, 39])

# Full attention (GQA)
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 256
ROTARY_DIMS = 64          # HEAD_DIM * partial_rotary_factor (256 * 0.25)
ROPE_THETA = 10_000_000.0

# Linear attention (GatedDeltaNet)
LINEAR_NUM_KEY_HEADS = 16
LINEAR_KEY_HEAD_DIM = 128
LINEAR_NUM_VALUE_HEADS = 32
LINEAR_VALUE_HEAD_DIM = 128
LINEAR_CONV_KERNEL = 4    # short causal conv kernel size

# MoE
NUM_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
MOE_INTERMEDIATE_SIZE = 512
SHARED_EXPERT_INTERMEDIATE_SIZE = 512

# Special tokens.
# Qwen has two stop tokens for chat: <|endoftext|>=248044 and <|im_end|>=248046.
# In chat mode the model ends its turn with <|im_end|>, so we must stop on either.
EOS_TOKEN_IDS = frozenset({248044, 248046})
EOS_TOKEN_ID = 248044  # kept for back-compat with anything still importing this
