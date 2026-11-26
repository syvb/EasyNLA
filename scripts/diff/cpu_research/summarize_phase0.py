"""Print the Gate 0 numbers from one or more phase0_report.json files as compact rows.

    python summarize_phase0.py <dir-or-json> [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

for arg in sys.argv[1:]:
    p = Path(arg)
    p = p / "phase0_report.json" if p.is_dir() else p
    r = json.loads(p.read_text())
    a, g = r["alignment"], r["gate0"]
    print(f"\n## {p.parent}")
    print(f"web-val FVE: identity={a['fve_identity']:.4f} id+bias={a['fve_identity_bias']:.4f} "
          f"std_ridge={a['fve_std_ridge']:.4f} (alpha={a['best_alpha_std_ridge']:g}) "
          f"id_anchored={a['fve_id_anchored']:.4f} (alpha={a['best_alpha_id_anchored']:g})")
    print(f"tau_web_median={a['tau_web_median']:.3f} p75={a['tau_web_p75']:.3f} "
          f"norm stats={ {k: round(v, 2) for k, v in a['delta_norm_stats'].items()} }")
    for w, wr in r["wrappers"].items():
        s = wr["auditing_summary"]
        t1 = wr["tok1_ridge_fve_user_content"]
        tag = " [WINNER]" if w == g["winning_wrapper"] else ""
        print(f"- {w}{tag}: KL={wr['neutral_kl_inst_vs_base_nats']:.3f} nats, amp={wr['chat_template_amplification']:.3f}, "
              f"SNR ratio mean={s['avg_fisher_snr_ratio']:.3f} min={s['min_task_snr_ratio']:.3f}, "
              f"tok1 h_t={t1['h_target']:.3f} d_raw={t1['delta_raw']:.3f} d_cu={t1['delta_role_centered_unit']:.3f} "
              f"(cut {t1['reduction_frac_vs_h_target'] * 100:.1f}%)")
        for task, tm in wr["tasks"].items():
            f, au = tm["fisher_snr"], tm["probe_auroc_5fold"]
            print(f"    {task:15s} SNR h_t={f['h_target']:.3f} d_raw={f['delta_raw']:.3f} d_cu={f['delta_role_centered_unit']:.3f} "
                  f"ratio={f['ratio_centered_unit_over_h_target']:.2f} | AUROC h_t={au['h_target']:.3f} "
                  f"d_cu={au['delta_role_centered_unit']:.3f} | mean-offset={tm['mean_offset_energy_frac']:.3f}")
    tb = r.get("taboo_mode_check", {})
    for lid, lv in (tb.get("loras") or {}).items():
        print(f"- taboo {lid}: KL no-think={lv['kl_enable_thinking_false']:.3f} think={lv['kl_enable_thinking_true']:.3f} "
              f"-> {lv['recommended_mode']} (passes<3: {lv['passes_kl_lt_3']})")
    print(f"GATE 0: passed={g['passed']} | fix1_best={g['fix1_best_passed']} fix1_strict={g['fix1_strict_passed']} "
          f"snr>=1.25x={g['snr_boost_ge_1_25x']} tok1_cut>=30%={g['tok1_cut_ge_30pct']} taboo={g['taboo_kl_lt_3']} "
          f"(taboo skipped={tb.get('skipped')})")
