# LaCO — Algorithm notes

LaCO (Layer Collapse) merges rear layers into a prior layer using RDSC:

θ*_l = θ_l + Σ_{k=1..m} (θ_{l+k} - θ_l) = θ_l + Σ θ_{l+k} - m * θ_l

Acceptance: after producing temporary merged model M_tmp, evaluate on few-shot calibration set D;
extract last-layer representations for each input, compute avg cosine similarity s = Avg_Cos_Sim(M_tmp, M, D).
If s > T, accept the merge; otherwise reject.

Partial merges:
- `full`: merge all parameters of the layers.
- `attn`: only apply merges to attention submodule parameters.
- `ffn`: only apply merges to MLP/ffn parameters.

Block parameters:
- C: number of layers to consider per merge attempt (max merged layers = C-1)
- L,H: search range
- I: minimal spacing between accepted merges
- T: similarity threshold

Tips to get good results:
- Use a few high-quality calibration sentences (5–20) representative of task domain.
- Start conservative T (0.995–0.999), relax to 0.99 if you want larger pruning ratio.
- For large LLMs, attn-only or ffn-only merges can preserve different capabilities — run ablations.

