"""Print the report's Markdown tables straight from the result JSONs, so no number is hand-copied.

    python make_report_tables.py <results_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
AUDIT = ("refusal", "eval_awareness", "sycophancy")


def load(name: str) -> dict | None:
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}"


p0 = load("phase0_report_8b_L24.json")
if p0:
    a = p0["alignment"]
    print("### Gate 0, Qwen3-8B L24 (repo script, runbook arguments)\n")
    print("| Base wrapper | Neutral KL (nats) | Refusal | Eval-awareness | Sycophancy | Mean (gate >= 1.25) | Min (gate >= 0.85) | tok1 cut |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for w, wr in p0["wrappers"].items():
        s = wr["auditing_summary"]
        r = s["per_task_snr_ratios"]
        win = " (chosen)" if w == p0["gate0"]["winning_wrapper"] else ""
        print(f"| {w}{win} | {f(wr['neutral_kl_inst_vs_base_nats'])} | {f(r['refusal'])} | {f(r['eval_awareness'])} | "
              f"{f(r['sycophancy'])} | {f(s['avg_fisher_snr_ratio'])} | {f(s['min_task_snr_ratio'])} | "
              f"{wr['tok1_ridge_fve_user_content']['reduction_frac_vs_h_target'] * 100:.0f}% |")
    print(f"\nweb FVE: identity {f(a['fve_identity'], 4)}, id+bias {f(a['fve_identity_bias'], 4)}, std ridge "
          f"{f(a['fve_std_ridge'], 4)}, id-anchored {f(a['fve_id_anchored'], 4)}; tau {f(a['tau_web_median'])}; gate {p0['gate0']}\n")

for tag, name in (("stock alignment", "gate_controls_8b_L24.json"), ("robust alignment", "gate_controls_8b_L24_robust.json")):
    gc = load(name)
    if not gc:
        continue
    print(f"### Gate statistic with controls, {gc['n_pairs']} pairs, {tag}\n")
    print("| Label | Kind | Base wrapper | SNR(h_target) | unit(h_target - mean) | h_base | unit(diff - mean) [95% CI] | 4-pair accuracy h_target -> diff |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for w in ("plain", "chatml"):
        for task, r in gc["wrappers"][w].items():
            lo, hi = r["ratio_ci95"]["delta_idanc_cu"]
            fs = r["few_shot_bal_acc_k4"]
            print(f"| {task} | {r['kind']} | {w} | {f(r['snr']['h_target'], 3)} | {f(r['ratio_vs_h_target']['h_target_cu'])} | "
                  f"{f(r['ratio_vs_h_target']['h_base'])} | {f(r['ratio_vs_h_target']['delta_idanc_cu'])} [{f(lo)}, {f(hi)}] | "
                  f"{f(fs['h_target'], 3)} -> {f(fs['delta_idanc_cu'], 3)} |")
    print()

dec = load("snr_decomposition_8b.json")
if dec:
    print("### Why the ratio is below 1 (stock alignment; all relative to h_target)\n")
    print("| Task | Base wrapper | Signal already in Base | Signal left in diff | Within-class spread of diff | SNR ratio |")
    print("| --- | --- | --- | --- | --- | --- |")
    for key, row in dec.items():
        task, w = key.split("/")
        s = row["stock"]
        print(f"| {task} | {w} | {f(s['between_base_over_target'])} | {f(s['between_delta_over_target'])} | "
              f"{f(s['within_delta_over_target'])} | {f(s['snr_delta_over_snr_target'])} |")
    print()

ra = load("robust_alignment_summary_8b.json")
if ra:
    print("### Alignment with and without massive-activation tokens\n")
    print(f"masked: train {ra['n_masked_train']}/{ra['n_train']}, val {ra['n_masked_val']}/{ra['n_val']} (norm > {ra['norm_mult']}x median)\n")
    print("| Map | Web-val FVE, all tokens | Web-val FVE, ordinary tokens | Median residual norm (tau), ordinary tokens |")
    print("| --- | --- | --- | --- |")
    allv, mv, tv = ra["val_fve_all_tokens"], ra["val_fve_masked_tokens"], ra["tau_median_masked_val"]
    print(f"| Identity | {f(allv['identity'], 3)} | {f(mv['identity'], 3)} | {f(tv['identity'], 1)} |")
    print(f"| Id-anchored ridge, as fitted by Phase 0 | {f(allv['stock_id_anchored'], 3)} | {f(mv['stock_id_anchored'], 3)} | {f(tv['stock_id_anchored'], 1)} |")
    print(f"| Id-anchored ridge, massive tokens masked | {f(allv['robust_id_anchored'], 3)} | {f(mv['robust_id_anchored'], 3)} | {f(tv['robust_id_anchored'], 1)} |")
    print(f"| Standard ridge, massive tokens masked | {f(allv['robust_std_ridge'], 3)} | {f(mv['robust_std_ridge'], 3)} | |")
    print()


def sweep_rows(sw: dict) -> list[dict]:
    """Per layer: mean audit / content ratio of unit(naive diff - mean), and of h_base, per Base wrapper."""
    out = []
    for li in range(sw["n_layers"]):
        row = {"layer": li}
        for w in ("chatml", "plain"):
            rw = sw["layers"][str(li)][w]
            row[w] = {
                "audit": rw["mean_audit_ratio"],
                "content": rw["mean_content_ratio"],
                "h_base": sum(rw[n]["ratio_h_base"] for n in AUDIT) / len(AUDIT),
                "h_target_cu": sum(rw[n]["ratio_h_target_cu"] for n in AUDIT) / len(AUDIT),
                "fs_t": sum(rw[n]["few_shot_k4_h_target"] for n in AUDIT) / len(AUDIT),
                "fs_d": sum(rw[n]["few_shot_k4_naive_diff_cu"] for n in AUDIT) / len(AUDIT),
            }
        out.append(row)
    return out


for tag, name, where in (("Qwen3-8B", "layer_sweep_8b.json", "`asst_last`"), ("Qwen3-0.6B", "layer_sweep_06b.json", "`asst_last`"),
                         ("Qwen3-8B", "layer_sweep_8b_user_last.json", "the last user token")):
    sw = load(name)
    if not sw:
        continue
    print(f"### Layer sweep, {tag}: SNR ratio vs raw h_target at {where}, naive diff, {sw['n_pairs']} pairs\n")
    print(f"agreement with the split loader: {sw['agreement_with_cached_io']}\n")
    print("| Layer | unit(h_target - mean) | ChatML: diff, audit | ChatML: diff, content | ChatML: h_base, audit | "
          "Plain: diff, audit | Plain: diff, content | Plain: h_base, audit | 4-pair accuracy h_target -> diff (ChatML / Plain) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in sweep_rows(sw):
        c, p = row["chatml"], row["plain"]
        print(f"| {row['layer']} | {f(c['h_target_cu'])} | {f(c['audit'])} | {f(c['content'])} | {f(c['h_base'])} | "
              f"{f(p['audit'])} | {f(p['content'])} | {f(p['h_base'])} | {f(c['fs_t'], 3)} -> {f(c['fs_d'], 3)} / {f(p['fs_d'], 3)} |")
    print()

ROLES = ("user_content", "asst_last", "asst_rollout")

for tag, name in (("stock alignment", "corpus_geometry_8b_stock.json"), ("robust alignment", "corpus_geometry_8b_robust.json")):
    cg = load(name)
    if not cg:
        continue
    win = cg["winning_wrapper"]
    other = "plain" if win == "chatml" else "chatml"
    print(f"### Arm 2 targets on the chat corpus, {tag} ({cg['n_convs']} conversations, tau {f(cg['tau_web_median'], 1)}, Base wrapper {win})\n")
    print("| Role | Rows | Massive rows | Within-role variance of h_target explained by aligned Base | Cleaned diff norm / tau (p10, median, p90) | "
          "Rows below tau | Fix 3 weight (min, median, max) | Effective sample size | Cosine of ChatML and Plain diffs |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for role in ROLES:
        r = cg["roles"].get(role)
        if not r:
            continue
        nf, q, w = r["noise_floor"], r["noise_floor"]["delta_centered_over_tau"], r["noise_floor"]["soft_norm_w"]
        print(f"| `{role}` | {r['n']} | {r['massive_rows']['n']} | {f(r['base_explains_target_within_role_fve'][f'id_anchored_{win}'])} | "
              f"{f(q['0.1'])}, {f(q['0.5'])}, {f(q['0.9'])} | {nf['frac_below_tau'] * 100:.0f}% | "
              f"{f(w['min'])}, {f(w['median'])}, {f(w['max'])} | {nf['soft_norm_ess_frac'] * 100:.0f}% | "
              f"{f(r['wrapper_agreement']['cos_delta_chatml_vs_plain'])} |")
    print()
    print(f"### Held-out share of each arm's unit target that a linear map recovers, {tag}\n")
    print("| Role | Target | From an independent reader LM on the same text | From h_target | From h_base | tok1 (token identity) | Participation ratio |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for role in ROLES:
        r = cg["roles"].get(role)
        if not r:
            continue
        for t in ("arm0_h_target", "arm1_naive_std", "arm2_cleaned"):
            rec = r["recoverability"][t]
            tok1 = r["composition"].get("tok1_fve", {}).get(t)
            print(f"| `{role}` | {t} | {f(rec['reader_lm']['fve'])} | {f(rec['h_target']['fve'])} | {f(rec['h_base_win']['fve'])} | "
                  f"{'' if tok1 is None else f(tok1)} | {r['composition']['spectrum'][t]['participation_ratio']:.0f} |")
    print()
    beh = cg["behaviour_refusal_asst_last"]
    print(f"### Refusal behaviour at `asst_last`, {tag}: {beh['label_counts_safety']} on safety prompts\n")
    if "fisher_snr" in beh:
        print("| Representation | Fisher SNR | Ratio to h_target | Grouped-CV AUROC | 8-example accuracy |")
        print("| --- | --- | --- | --- | --- |")
        for k_, v in beh["fisher_snr"].items():
            print(f"| {k_} | {f(v, 3)} | {f(v / beh['fisher_snr']['h_target'])} | {f(beh['grouped_auroc'][k_], 3)} | "
                  f"{f(beh['few_shot_bal_acc_k8'][k_], 3)} |")
    print()

for tag, name in (("Qwen3-8B L24", "alignment_variants_8b.json"), ("Qwen3-0.6B L18", "alignment_variants_06b.json")):
    av = load(name)
    if not av:
        continue
    print(f"### Gate statistic under other alignments, {tag} ({av['n_pairs']} pairs; cells are ratio [95% CI])\n")
    print("| Base wrapper | Alignment | Refusal | Eval-awareness | Sycophancy | Mean audit | Mean content control | Held-out FVE at chat positions (user, last, rollout) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for w in ("plain", "chatml"):
        for v, r in av["wrappers"][w].items():
            cells = [f"{f(r['tasks'][n]['ratio'])} [{f(r['tasks'][n]['ci95'][0])}, {f(r['tasks'][n]['ci95'][1])}]" for n in AUDIT]
            fve = r.get("heldout_within_role_fve", {})
            print(f"| {w} | {v} | " + " | ".join(cells) + f" | {f(r['mean_audit_ratio'])} | {f(r['mean_content_ratio'])} | "
                  + ", ".join(f(fve[x]) for x in ROLES if x in fve) + " |")
    print()

for tag, name in (("Qwen3-8B L24, stock alignment", "behaviour_link_8b_stock.json"),
                  ("Qwen3-8B L24, robust alignment", "behaviour_link_8b_robust.json"),
                  ("Qwen3-0.6B L18", "behaviour_link_06b.json")):
    bl = load(name)
    if not bl:
        continue
    win = bl["winning_wrapper"]
    print(f"### Does the cleaned diff track KL(Instruct || Base)? {tag}, Base wrapper {win}\n")
    print(f"fp16 residual check: {bl['fp16_feed_check']}\n")
    print("| Role | Rows | Median KL (nats) | Top-1 agreement | Spearman: diff norm vs KL | Spearman: centred h_target norm vs KL | "
          "Fix 3 weight on top-quartile-KL rows (25% if uniform) | R2 of log KL from arm 2 target | from centred h_target | from h_base | "
          "Arm 2 target FVE from both next-token states | Arm 0 target, same |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for role in ROLES:
        r = bl["roles"].get(role)
        if not r:
            continue
        kl, sp, r2, tf = r["kl_inst_base_nats"][win], r["spearman_with_kl"][win], r["log_kl_r2_from"], r["target_fve_from_next_token_state"]
        print(f"| `{role}` | {r['n']} | {f(kl['median'])} | {kl['top1_agree'] * 100:.0f}% | {f(sp['cleaned_diff_norm'])} | "
              f"{f(sp['h_target_centered_norm'])} | {r['fix3_weight_share_on_top_quartile_kl'] * 100:.0f}% | {f(r2['arm2_cleaned'])} | "
              f"{f(r2['arm0_h_target_centered'])} | {f(r2['h_base_win'])} | {f(tf['arm2_cleaned']['final_inst+final_base_win'])} | "
              f"{f(tf['arm0_h_target']['final_inst+final_base_win'])} |")
    print()

for tag, name in (("Qwen3-8B L24", "cross_patch_8b.json"), ("Qwen3-0.6B L18", "cross_patch_06b.json")):
    cp = load(name)
    if not cp:
        continue
    print(f"### Where the behavioural gap lives, {tag} (ChatML sequences; KL from Instruct in nats, mean)\n")
    print("| Role | Rows | Full gap: Base | Base upper blocks on Instruct residuals | Instruct upper blocks on Base residuals | "
          "Gap left when the residual stream is Instruct's | Positions where that closes half the gap |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for role in ROLES:
        r = cp["roles"].get(role)
        if not r:
            continue
        k_ = r["kl_nats"]
        print(f"| `{role}` | {r['n']} | {f(k_['full_gap']['mean'])} | {f(k_['top_swapped']['mean'])} | {f(k_['bottom_swapped']['mean'])} | "
              f"{r['gap_left_with_instruct_residuals'] * 100:.0f}% | {r['frac_positions_residual_swap_closes_half_the_gap'] * 100:.0f}% |")
    print()

tm = load("taboo_mode_check_8b.json")
if tm:
    print("### Taboo mode check, Qwen3-8B L24\n")
    print(f"Instruct top-1 at `asst_last`: no-think {tm['instruct']['neutral/nothink']}, think {tm['instruct']['neutral/think']}\n")
    print("| Organism | KL no-think | KL think at `asst_last` | KL think at first reasoning token | Phase 4 rule picks | "
          "Org - Inst norm / tau (no-think, think) | Cosine of mean Org - Inst, think vs no-think |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for org, r in tm["organisms"].items():
        d = r["org_minus_inst_asst_last"]
        print(f"| {org} | {f(r['neutral/nothink']['kl_org_inst_nats'])} | {f(r['neutral/think']['kl_org_inst_nats'], 4)} | "
              f"{f(r['neutral/think_content']['kl_org_inst_nats'])} | {'thinking' if r['phase4_rule_picks_thinking'] else 'no-think'} | "
              f"{f(d['nothink']['median_norm_over_tau'])}, {f(d['think']['median_norm_over_tau'])} | {f(d['cos_mean_diff_think_vs_nothink'])} |")
    a = tm["instruct"]["asst_last_think_vs_nothink"]
    print(f"\nInstruct `asst_last`, think vs no-think: distance {f(a['median_distance_over_tau'])} tau, cosine {f(a['mean_cos'])}\n")

for tag, name in (("Qwen3-8B", "gate_positions_8b.json"), ("Qwen3-0.6B", "gate_positions_06b.json")):
    gp = load(name)
    if not gp:
        continue
    print(f"### Gate statistic by position, {tag} L{gp['layer']} ({gp['n_pairs']} pairs; SNR ratio unit(x - mean) / raw h_target [95% CI])\n")
    print("| Alignment | Base wrapper | Position | Refusal | Eval-awareness | Sycophancy | Mean audit | Mean content control |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for kind, r in gp["no_base"].items():
        cells = [f"{f(r[n]['ratio'])} [{f(r[n]['ci95'][0])}, {f(r[n]['ci95'][1])}]" for n in AUDIT]
        content = [v["ratio"] for v in r.values() if v["kind"] == "content_control"]
        print(f"| none (h_target only) | | {kind} | " + " | ".join(cells)
              + f" | {f(sum(r[n]['ratio'] for n in AUDIT) / 3)} | {f(sum(content) / len(content))} |")
    for a, ra_ in gp["alignments"].items():
        for key, r in ra_.items():
            w, kind = key.split("/")
            cells = [f"{f(r[n]['ratio'])} [{f(r[n]['ci95'][0])}, {f(r[n]['ci95'][1])}]" for n in AUDIT]
            print(f"| {a} | {w} | {kind} | " + " | ".join(cells) + f" | {f(r['mean_audit_ratio'])} | {f(r['mean_content_ratio'])} |")
    print("\nRaw h_target Fisher SNR by position: " + "; ".join(
        f"{kind}: " + ", ".join(f"{n} {f(r[n]['snr_h_target'], 3)}" for n in AUDIT) for kind, r in gp["no_base"].items()) + "\n")
    print("Why, per task (refusal / eval-awareness / sycophancy), all relative to raw h_target at the same position:\n")
    print("| Alignment | Base wrapper | Position | SNR of Base alone | Class signal kept by the diff | Within-class spread of the diff | Diff Fisher SNR, absolute |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for a, ra_ in gp["alignments"].items():
        for key, r in ra_.items():
            w, kind = key.split("/")
            if "snr_h_base_over_h_target" not in r[AUDIT[0]]:
                continue
            print(f"| {a} | {w} | {kind} | " + " / ".join(f(r[n]["snr_h_base_over_h_target"]) for n in AUDIT) + " | "
                  + " / ".join(f(r[n]["signal_kept"]) for n in AUDIT) + " | " + " / ".join(f(r[n]["within_spread"]) for n in AUDIT)
                  + " | " + " / ".join(f(r[n]["snr_rep"], 3) for n in AUDIT) + " |")
    print()

cb = load("corpus_behaviour_positions_8b.json")
if cb:
    print(f"### Refuse vs comply on the model's own rollouts, by role ({cb['n_refuse']} refusals of {cb['n']} safety conversations)\n")
    names = list(next(iter(cb["roles"].values())))
    print("| Role | SNR(h_target) | " + " | ".join(names) + " |")
    print("| --- | --- | " + " | ".join("---" for _ in names) + " |")
    for role, r in cb["roles"].items():
        print(f"| `{role}` | {f(r['no_base']['snr_h_target'], 3)} | " + " | ".join(
            f"{f(r[n]['ratio'])} [{f(r[n]['ci95'][0])}, {f(r[n]['ci95'][1])}]" for n in names) + " |")
    print()

dwe = load("dw_coordinate_energy.json")
if dwe:
    print("### Where the identity-anchored correction writes\n")
    print("| Alignment | Largest output coordinate | Its share of the map's energy | Coordinates holding half the energy |")
    print("| --- | --- | --- | --- |")
    for name, r in dwe.items():
        t = r["top_output_coordinates"][0]
        print(f"| {name} | {t['coordinate']} | {t['share'] * 100:.1f}% | {r['coordinates_for_half_energy']} |")
    print()

for tag, name in (("stock", "top_direction_8b_stock.json"), ("robust", "top_direction_8b_robust.json")):
    td = load(name)
    if not td:
        continue
    win = td["winning_wrapper"]
    print(f"### Largest direction of the Arm 2 unit target, {tag} alignment\n")
    print("| Role | Variance share of top direction: h_target | fitted diff | identity diff | Participation ratio: h_target | fitted diff | identity diff | "
          "Largest coordinate | Its share of the direction | Spearman with KL(Instruct, Base) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for role, r in td["roles"].items():
        sp, t = r["spectrum"], r["top_direction"]
        kl_ = t["spearman_with"].get("kl_inst_base")
        print(f"| `{role}` | {sp['h_target']['top3_share'][0] * 100:.0f}% | {sp[f'idanc_{win}']['top3_share'][0] * 100:.0f}% | "
              f"{sp[f'identity_{win}']['top3_share'][0] * 100:.0f}% | {sp['h_target']['participation_ratio']:.0f} | "
              f"{sp[f'idanc_{win}']['participation_ratio']:.0f} | {sp[f'identity_{win}']['participation_ratio']:.0f} | "
              f"{t['largest_coordinate']} | {t['largest_coordinate_share_of_direction'] * 100:.0f}% | {'' if kl_ is None else f(kl_)} |")
    print()

for tag in ("stock", "robust"):
    tt = load(f"taboo_transfer_8b_{tag}.json")
    if not tt:
        continue
    tn = load(f"taboo_norm_vs_training_8b_{tag}.json")
    print(f"### Taboo Org - Inst diffs, no-think, {tag} alignment (L{tt['layer']}, {tt['n_prompts']} prompts x {len(tt['organisms'])} organisms)\n")
    print("| Position | Median norm / tau (min to max over organisms) | Training rows: 5th, 50th, 95th percentile norm / tau | "
          "Training rows smaller than the Taboo diff | Soft-norm injection strength: Taboo, median training row | Energy in the shared direction | "
          "Organism ID, 10-way | Word ID across regimes, 5-way (narrow to mix, mix to narrow) | "
          "Top-10 share: Taboo diff, held-out training target | Top-100 share: same |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for kind, r in tt["positions"].items():
        norms = sorted(r["median_norm_over_tau"].values())
        sh = r["vs_arm2_training_target"]
        n_ = tn["positions"][kind] if tn else None
        q = ", ".join(f(x) for x in n_["train_norm_over_tau_q05_q50_q95"]) if n_ else ""
        print(f"| {kind} | {f(float(sum(norms[4:6]) / 2))} ({f(norms[0])} to {f(norms[-1])}) | {q} | "
              + (f"{n_['train_frac_below_taboo_median'] * 100:.1f}% | {f(n_['soft_norm_strength_taboo'])}, {f(n_['soft_norm_strength_train_median'])}" if n_ else " | ")
              + f" | {r['shared_direction_energy_share'] * 100:.0f}% | {r['organism_id_acc_10way'] * 100:.1f}% | "
              f"{r['word_id_acc_5way_narrow_to_mix'] * 100:.0f}%, {r['word_id_acc_5way_mix_to_narrow'] * 100:.0f}% | "
              + " | ".join(f"{f(sh[f'top{kk}_share']['taboo_org_minus_inst'])}, {f(sh[f'top{kk}_share']['heldout_training_targets'])}" for kk in (10, 100)) + " |")
    print()
    if tag != "stock":
        continue
    print("Logit lens on the diff (Instruct's final-norm gain and unembedding, no upper blocks):\n")
    print("| Position | Mean removed over organisms | Mean diff: secret word first of 5 | Single prompts: secret word first of 5 (chance 20%) |")
    print("| --- | --- | --- | --- |")
    for kind, r in tt["positions"].items():
        for c in ("uncentred", "centred"):
            print(f"| {kind} | {'yes' if c == 'centred' else 'no'} | {r[f'logit_lens_first_of_5_{c}'] * 100:.0f}% | "
                  f"{r[f'logit_lens_single_prompt_first_of_5_{c}'] * 100:.1f}% |")
    print()
    print("| Organism | user_mean, uncentred: vocabulary rank of the secret word | single-prompt median rank | single prompts first of 5 | top tokens | "
          "asst_last, uncentred: vocabulary rank | single prompts first of 5 | top tokens |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    um, al = tt["positions"]["user_mean"]["logit_lens"], tt["positions"]["asst_last"]["logit_lens"]
    for org in tt["organisms"]:
        a, b = um[org]["uncentred"], al[org]["uncentred"]
        print(f"| {org} | {a['vocab_rank_of_secret_word']} | {a['single_prompt_median_vocab_rank']:.0f} | {a['single_prompt_frac_first_of_5'] * 100:.0f}% | "
              f"{', '.join(t.strip() for t in a['top_tokens'][:5])} | {b['vocab_rank_of_secret_word']} | {b['single_prompt_frac_first_of_5'] * 100:.0f}% | "
              f"{', '.join(t.strip() for t in b['top_tokens'][:4])} |")
    print()

for tag, name in (("Qwen3-8B", "massive_anatomy_8b.json"), ("Qwen3-0.6B", "massive_anatomy_06b.json")):
    ma = load(name)
    if not ma:
        continue
    print(f"### Massive activations, {tag} L{ma['layer']}\n")
    print("| Tokens | Web train: massive after position 0 (Target, Base) | Both fire, Target only, Base only | Largest ordinary norm / median (Target, Base) | "
          "Smallest massive norm / median | Share of energy in the top coordinate | Share of h_target variance | Share of the alignment objective |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    w = ma["web"]["train"]
    t_, b_ = w["target"], w["base"]
    if "residual" in w:
        fr = w["fires"]
        print(f"| {w['n_tokens']} | {t_['n_massive']}, {b_['n_massive']} | {fr['both']}, {fr['target_only']}, {fr['base_only']} | "
              f"{f(t_['largest_ordinary_norm_over_median'])}, {f(b_['largest_ordinary_norm_over_median'])} | "
              f"{f(t_['smallest_massive_norm_over_median'], 1)}, {f(b_['smallest_massive_norm_over_median'], 1)} | "
              f"{max(t_['top_coordinates_share_of_massive_energy'].values()) * 100:.1f}% | {t_['share_of_centred_variance_in_massive_rows'] * 100:.0f}% | "
              f"{w['residual']['share_of_centred_squared_residual_in_massive_rows'] * 100:.1f}% |")
        print(f"\nTokens: {w['tokens'][:5]}; rows masked by threshold (x median): {w['n_masked_by_threshold']}; "
              f"documents with one: {w['documents_with_a_massive_token']} of {w['n_documents']}\n")
    else:
        print(f"| {w['n_tokens']} | {t_['n_massive']}, {b_['n_massive']} | | | | | | |\n")
    for key, c in ma.get("chat", {}).items():
        print(f"- chat, {key}: {c['n_massive_at_position_0']} at position 0, {c['n_massive'] - c['n_massive_at_position_0']} later of {c['n_tokens']} tokens, "
              f"{c['n_massive_at_a_position_phase2_samples']} at a position Phase 2 samples")
    print()

mv = load("massive_fix_variants_8b.json")
if mv:
    print(f"### Fixes for the alignment fit, Qwen3-8B L{mv['layer']} (alpha {mv['alpha']}; ordinary = not massive in either model)\n")
    print("| Variant | FVE, ordinary val tokens | tau | Share of the correction in its largest coordinate | Diff std in the massive coordinate, ordinary tokens | "
          "Its share of the diff's variance | A row where one model fires, norm / tau | Where both fire | Arm 2 target participation ratio (user, rollout) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, r in mv["variants"].items():
        print(f"| {name} | {f(r['fve_ordinary_val'], 3)} | {f(r['tau_median_ordinary_val'], 1)} | {r['dW_share_in_largest_output_coordinate'] * 100:.1f}% | "
              f"{f(r['ordinary_val_diff_std_in_massive_coordinate'], 1)} | {r['ordinary_val_share_of_diff_energy_in_massive_coordinate'] * 100:.1f}% | "
              f"{f(r['one_model_fires_val_median_diff_norm_over_tau'], 1)} | {f(r['both_fire_val_median_diff_norm_over_tau'], 1)} | "
              f"{r['arm2_target_user_content']['participation_ratio']:.0f}, {r['arm2_target_asst_rollout']['participation_ratio']:.0f} |")
    print()
