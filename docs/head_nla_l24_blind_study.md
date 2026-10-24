# Blind study: do matryoshka explanations and mechanistic analysis agree about a head?

**Heads:** Qwen3-8B layer 24, heads **H29**, **H30** and **H13**. They were chosen for high to moderate matryoshka FVE (0.63, 0.57, 0.40), and none had been analysed before. H29 and H30 share a KV head (H28–31), and H13 shares one with H12–15.

**Setup** (2026-10-23; data in `~/blind_study/`, public on HF in `syvb/head-nla-qwen3-8b` under `l24/blind_study/`). Two agents worked independently, each with its own data bundle, and neither could read the repo docs, earlier results or memory.

- **Explanations arm.** It had only the matryoshka AV's explanations, shuffled: about 1,240 per head, from every token of the 4 doc-viewer documents plus about 60 held-out tokens per head from the 2,000-row sweep. It also had 60 explanations from each of the other 29 heads for contrast. It saw no tokens, positions, attention, scores or weights.
- **Mechanistic arm.** It had the 4 documents' tokens, full layer-24 attention, the heads' writes, per-token ablation effects and vocabulary readouts of the writes (lens, avg16), plus the layer-24, final-norm and unembedding weights (CPU only). It saw no explanations.
  - Its main method was to recover each attended token's value contribution by inverting attention against the writes, then read those values out in vocabulary space through each head's own output weights.
  - Check: heads sharing a KV head recovered the same per-source values (r = 0.94–0.97), even though each was solved separately.
- **Blind judge.** A third agent got the six ≤120-word summaries and key claims, with head numbers removed and the order shuffled within each set. It scored all 9 explanation-vs-mechanistic pairs from 1 to 5 and gave a one-to-one matching.

## Accounts

| Head | Explanations arm | Mechanistic arm |
|---|---|---|
| **H13** | Carries the *kind* of entity or means in play, and it tracks the document's topic: tools, engines, software; treatments and drugs; women, victims, perpetrators, owners, agencies. Often framed as "the hidden one behind it". | Reads content words about the passage's main entities ("dead", "found", "jailed", "LLVM", "treatment", "antidepressants"). Writes a generic **category noun** for that entity: woman/female, this person, officials, companies/owners, tool/platform, treatment method, medicines. Largest effects: "Pate was a wonderful" → " woman" (+0.43 nats) and "no regard for her" → " employers". |
| **H29** | Headline: a near-constant "conditional doubt / negation / absence" stance (60% of sites). Minority: letter-like identity "O, L, T, P" (about 3%), women's names ("Rose"). | Mostly parked on the first token (sink). Leaves it before a noun phrase where a name already seen could recur. Then attends to the end of that earlier name and writes its **first token or initial letter** (Ocoee → ' O', TMS → ' T', Lancer Circle → ' L', Lisa). Small, sparse effects. |
| **H30** | Headline: "conditional skepticism / speculation" stance (46% of sites). Minority: **agents and sources**: reporters, audiences, authorities, named people. | Same keys and values as H29, but it reads "said", "is", "was" and punctuation, whose values carry their subject. Writes the **speaker's or subject's identity** (' Kelly', ' Ms', 'Officials', initials). Clearest at closing quotes before an attribution. |

## Blind judge

Compatibility scores (1–5); the correct pairs are on the diagonal:

| | mech H13 | mech H29 | mech H30 |
|---|---|---|---|
| expl H13 | **5** | 2 | 2 |
| expl H29 | 1 | **3** | 2 |
| expl H30 | 2 | 2 | **3** |

- **Matching: 3 of 3 correct.** Confidence 95% for H13, 78% each for H29 and H30.
- **H13:** an almost one-for-one match of specific claims.
- **H29 and H30:** the judge found that the explanations' *headlines* (doubt or skepticism) have no mechanistic counterpart. The matches were driven by their small-share secondary claims: letter identity for H29, sources and speakers for H30.

## The headline mismatch is the AV's reading of a sink-parked write

At each site, compare how much the head attends to the first token with how often the matryoshka explanation opens with doubt, hedging, negation or absence (doc-viewer data):

| Sink-attention quartile | H29: mean sink attention | H29: write norm | H29: share "doubt" | H30: mean sink attention | H30: share "doubt" | H13: mean sink attention | H13: share "doubt" |
|---|---|---|---|---|---|---|---|
| Lowest | 0.44 | 1.05 | **0.03** | 0.28 | **0.00** | 0.17 | **0.01** |
| 2nd | 0.75 | 0.44 | 0.49 | 0.59 | 0.13 | 0.39 | 0.02 |
| 3rd | 0.89 | 0.33 | 0.94 | 0.80 | 0.75 | 0.59 | 0.14 |
| Highest | 0.97 | 0.34 | **1.00** | 0.93 | **1.00** | 0.82 | **0.80** |

- Correlation between a "doubt" opening and sink attention: +0.78 for H29, +0.78 for H30, +0.69 for H13.
- So "conditional doubt" is what the AV says when the head is parked on the sink and its write is small. It is a label for "no content here", not a function. H29 and H30 are sink-parked most of the time, so this default dominated the explanations arm's statistics.
- **Where the heads do read real tokens, the explanations match the mechanistic account** (lowest-sink 10% of sites):
  - **H29:** "**L** and locations: … **the first letter** of a line, and references back to an L", "L=L shapes", "Rose or Rose", "Huang (names…)", "Compiler-style compiler".
  - **H30:** "**Reporter names** and eyewitness records", "Persons and **reporters names**: … a **speaker reference**", "T-Turning Tactics, **T is for T**…".

## Takeaways

1. **Agreement is good where a head writes real content.** Using matryoshka explanations alone, an analyst correctly identified H13's function (entity category) and the specific contents of H29 (name initials) and H30 (speakers and sources). A blind judge paired every head with its mechanistic account.
2. **The weakness is the null reading.** For heads that mostly park on the sink, the explanations' most frequent reading is a stock "doubt/hedging" stance. An analyst who aggregates over all sites will mistake it for the head's function. Weighting by write norm, or dropping sink-dominated sites, would fix this. So would an AV trained to say "nothing here" for near-null writes, which ties into the warm-start format ideas (an explicit Activity/Confidence field).
3. **The explanations do not show the mechanism:** which tokens the head reads (H30's "said" keys; H29 attending to the end of the earlier name) or when it fires. The mechanistic arm found that only by using attention, which the NLA never sees. Related: when quoted terms in the explanations appear in the context, the explaining head attends to them 3–5× more than other heads do (`scripts/heads_quote_attention.py`).
