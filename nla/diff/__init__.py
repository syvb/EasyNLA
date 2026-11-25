"""Direct Diff-Target NLA modules (Fixes 1-3, wrappers, metrics, benchmarks).

Submodules:
  - alignment:  Fix 1 — Identity-Anchored Ridge alignment (W = I + delta_W) & CV.
  - wrappers:   Fix 2 & §2.4 — ChatML / Base(Plain) 1-to-1 token alignment,
                template masking, and per-role mean centering.
  - metrics:    Geometry diagnostics (FVE, Fisher SNR, linear probe AUROC,
                tok1 Ridge leakage, next-token KL, verbatim overlap, bootstrap CI).
  - benchmarks: Diagnostic & Phase 4 auditing benchmark loaders + starter sets.
"""
