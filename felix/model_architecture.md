Model Overview

    Type: Causal Language Model with Vision Encoder
    Training Stage: Pre-training & Post-training
    Language Model
        Number of Parameters: 35B in total and 3B activated
        Hidden Dimension: 2048
        Token Embedding: 248320 (Padded)
        Number of Layers: 40
        Hidden Layout: 10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))
        Gated DeltaNet:
            Number of Linear Attention Heads: 32 for V and 16 for QK
            Head Dimension: 128
        Gated Attention:
            Number of Attention Heads: 16 for Q and 2 for KV
            Head Dimension: 256
            Rotary Position Embedding Dimension: 64
        Mixture Of Experts
            Number of Experts: 256
            Number of Activated Experts: 8 Routed + 1 Shared
            Expert Intermediate Dimension: 512
        LM Output: 248320 (Padded)
        MTP: trained with multi-steps
    Context Length: 262,144 natively and extensible up to 1,010,000 tokens.
