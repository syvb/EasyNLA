# L24 head 18 (and its KV group 16–19): a document-domain head

Case study from the doc-viewer data: 4 held-out docs, every token × every head, full attention. All of it runs on CPU.
Reproduce with:

```
python scripts/heads_kv_group_study.py --viewer-dir ~/l24_docs/hf/l24/std_rl/doc_viewer --head 18
```

The script needs three Qwen3-8B tensors: L24 `o_proj`, the final norm and `lm_head`.

**Why H18.** It is in the KV group with the highest NLA scores: H16–19 have RL FVE of +58…+63%.
Of all 32 heads, H18 has the best gold (Haiku-explanation) FVE, at +28%.

**Method.**
- Every "write" here is what the NLA sees: v = O_h(z − z̄)/rms, the head's write minus its average write over a large corpus. So the per-source values below are v_j − z̄. Attention rows sum to 1, so the shift is exact.
- z = pinv(O_h)·write recovers each head's 128-d attention output, minus z̄. The residual is 2e-4.
- Q heads 16–19 share one KV head. Stacking their four attention patterns gives an overdetermined system A·V = Z, which yields the per-source values v_j.
- That gives each source token's write O_h·v_j. The fit's relative error is 5–11%: fp16 attention, and bf16 SDPA in the model.

## What it does

1. **Sink = a constant "off" write.**
   - About 57% of H18's attention goes to token 0, on average.
   - The sink write is the same direction in every doc (cos ≥ 0.999).
   - Its lens is junk: `$\`, `invariant`, `exploits`, `unp`, … These tokens show up in every doc's per-site lens lists.
2. **Everything else = the document's domain.**
   - The values at non-sink positions carry a doc-level domain feature. Their logit lens reads:
     - homicide news: 公安机关 (police organ), 被害人 (victim), 民警
     - programming-languages blog: `LLVM`, `semantics`, `Turing`, `lexical`, `::=`
     - TMS / depression article: `clinical`, `clinicians`, 临床, `psychotic`
     - fraud court report: 被害人, 被告人 (defendant), 侦查 (investigation), 刑事
   - **Function tokens carry the domain too.** It is in the values of `.\n`, `The`, ` said`, `,`, ` the`, not only in the values of `Police`, `compiler` or `MS`. By L24 the residual stream at every position already encodes the domain, and this head copies it out.
3. **The write is ≈ 2-dimensional per doc.**
   - Sink direction + one doc-topic direction capture 75–90% of each write's direction (mean cos²).
   - Sink alone captures 14–37%. Sink plus *another* doc's topic captures 14–44%.
   - The two crime docs share a topic direction (cos 0.70); the other pairs are ≈ 0.
4. **Attention selects topic-heavy keys only weakly.**
   - The most-attended keys are topical: `Police`, `DS`, `jailed`, `compiler`, `LLVM`, `MS`, `imulation`.
   - But the rank correlation between the attention a key receives and its value's alignment with the doc topic is only +0.07…+0.21. Every value has the topic, so where attention lands doesn't change much.
   - **The query decides how much:** H18 writes topic most at function words (` her` 0.24 sink, ` as`, ` of`, ` an`, ` that`, ` with`, ` to`, ` a`), and least at digits, `?`, ` said`, ` at` and name fragments (0.7–0.9 sink).
   - Reading (a hypothesis): it adds a domain prior where a content word is about to be chosen.
5. **The KV group is a family with different readouts.** The heads share values, but each O_h reads a different side of the same domain feature (mean non-sink write lens):

| doc | H16 (roles) | H17 (events / mechanisms) | H18 (institutional register) | H19 (activity / field) |
|---|---|---|---|---|
| homicide | police, 警察, 派出所 | 涉嫌, 证据, suspected | 公安机关, 被害人, 民警 | 报警, 报案, 交通事故 |
| PL blog | 程序员, Computer, programmer | primitives, algorithms, syntax | Turing, semantics, LLVM | programming, software, bytecode |
| TMS | 医, 患者, clinical | neurological, serotonin, 认知 | clinical, clinicians, 临床 | 治疗, 抑郁症, Psychiatry |
| fraud | Criminal, 律师, 刑事 | 证据, 涉嫌, fraud, conspiracy | 被害人, 被告人, 侦查 | financial, 财务, theft |

## What the explanations get right

The table gives the share of H18 explanations whose text names each domain (regex). Docs 01 and 04 are crime, 02 is computing, 03 is medical; the bold cells are each doc's own domain.

| source | 01 crime | 02 computing | 03 medical | 04 crime | wrong-domain mentions |
|---|---|---|---|---|---|
| gold (Haiku) | **.54** | **.74** | **.82** | **.87** | ≤ .07 |
| RL | **.86** | **.98** | **.95** | **.97** | crime in 46% of PL-blog and 61% of TMS rows |
| SFT | .16 | .32 | .26 | .53 | low |

- **RL names the right domain almost always,** but it over-says "crime/legal" on the other docs.
- **Neither RL nor gold says what the head really is:** a constant sink write plus a doc-wide domain broadcast, gated by the query. Both describe it per site, as a "secondary pull toward X".
- **The NLA has to carry about two numbers per site** (domain identity, and sink-vs-topic mix), which is why this group is easy.
- **RL FVE is highest when the sink dominates** (+71% in the top sink quartile vs +58–62% elsewhere), because those writes are nearly the constant direction.

## Caveats

- 4 docs, 2 of them crime.
- The logit lens at L24 of 36 is only indicative. There is **no causal test**: ablating H16–19 and measuring the loss on domain words needs a GPU.
- The per-source values come from a noisy least-squares solve. Individual low-attention tokens (e.g. `Friday` with |u| = 10) can be artifacts; the aggregates are stable.
- The domain regexes are crude.

## Can the explanations tell the four sibling heads apart?

H16–19 read the same values at every site, but their writes are nearly orthogonal (mean cos 0.02–0.19). So the question is whether an explanation says anything specific to *its* head.

**Text classifier.** `scripts/heads_sibling_textclf.py` trains a TF-IDF logistic regression on 3 docs and tests on the 4th. It is 4-way, so chance is 25%.

| input | H16–19 | H0–3 | H24–27 |
|---|---|---|---|
| RL text | .55 | .71 | .43 |
| gold text | .59 | .72 | .30 |
| SFT text | .35 | .31 | .28 |
| lens readout | .63 | .81 | .35 |
| avg16 readout | .57 | .84 | .43 |

The words that separate the four heads' RL texts line up with the facets found by the logit lens. They also reflect each head's constant sink write:
- **H16** (roles; its sink lens is `Teacher`, `Classroom`): RL text uses medical, clinical, student(s), police, hospitals.
- **H17** (mechanisms): anxiety, brain, behavior, recursive, sequences, data structures.
- **H18** (constant junk sink write): vague, uncertainty, absence, modal. This is the hedging template.
- **H19** (activity; its sink lens is `mathematical`, `algebra`): therapy, depression, TMS, Lisp, languages, arithmetic, algebra, math.

**Blinded matching.** `scripts/heads_sibling_match.py` uses Sonnet 5.5.
- For each of 160 sites (40 per doc), Sonnet sees the 4 siblings' readouts (avg16 + lens), labelled A–D.
- It also sees the 4 siblings' explanations, shuffled, and assigns them one-to-one to the readouts.
- 617 of 640 calls returned a valid matching.

| source | accuracy (chance .25) |
|---|---|
| gold | 1.00 (ceiling: Haiku wrote these from the same readouts) |
| **RL** | **.79** (per head .72–.85; per doc .75–.88) |
| RL from another position of the same head (control) | .45 |
| SFT | .50 |

**Reading.**
- RL explanations do carry readable, head-specific information.
- Part of it is head-level (the shuffled-position control gets .45, from each head's constant sink write and style).
- The rest (.45 → .79) is specific to the site.
- This tests agreement with the readouts, not with the mechanism. The lens readouts alone separate the heads about as well (the classifier gets .63 from them), so the explanations mostly restate readout content in prose.
- The constant sink write, query gating and shared values still appear nowhere in the text.
