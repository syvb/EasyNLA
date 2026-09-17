"""Build a tiny, REAL NLA dataset for the pred-NLA CPU smoke test.

Not synthetic: activations come from an actual forward pass of a small model, the
marker token is chosen by the same auto-picker datagen uses, and the sidecar is
written through the same serializer. So the smoke test exercises the true
contract (marker ids, neighbour ids, prompt templates, d_model) rather than a
mock that can drift away from it.

    .venv/bin/python scripts/make_pred_smoke_data.py \
        --model Qwen/Qwen3-0.6B --layer 18 --out /tmp/pred_smoke/rl.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nla.datagen.injection_tokens import build_token_meta
from nla.datagen.sidecar import (
    NLADatasetMeta,
    NLAExtractionMeta,
    serialize_sidecar,
)
from nla.schema import INJECT_PLACEHOLDER, sidecar_path_for, wrap_explanation

# Distinct topics, deliberately: the shuffled-activation control pairs an
# explanation with ANOTHER document's continuation, so if the corpus were a few
# passages repeated under different ids, a "mismatched" explanation would often
# still be about the right subject and the control would read backwards.
_DOCS = [
    "The Apollo programme ran from 1961 to 1972 and put twelve people on the "
    "surface of the Moon. Its Saturn V rocket remains the most powerful launch "
    "vehicle ever flown operationally, and the guidance computer aboard the "
    "command module had less memory than a modern pocket calculator.",
    "Ocean acidification happens when seawater absorbs carbon dioxide from the "
    "atmosphere, lowering its pH. Shell-forming organisms such as oysters, "
    "corals and pteropods struggle to build calcium carbonate structures in "
    "more acidic water, which affects the whole food web above them.",
    "The Congress of Vienna redrew the map of Europe after the Napoleonic wars. "
    "Delegates spent months arguing over the disposition of Saxony and Poland, "
    "and the balance of power they settled on lasted, with interruptions, until "
    "the outbreak of war in 1914.",
    "In a compiler, the lexer turns a stream of characters into tokens and the "
    "parser turns those tokens into a syntax tree. Later passes lower that tree "
    "into an intermediate representation, run optimisations over it, and finally "
    "emit machine code for the target architecture.",
    "Sourdough bread relies on a culture of wild yeast and lactic acid bacteria. "
    "The bacteria produce acids that give the loaf its sour flavour and also "
    "strengthen the gluten network, which is why a well-fermented dough holds "
    "its shape better than one raised with commercial yeast alone.",
    "Mangrove forests grow in the tidal zone where fresh and salt water mix. "
    "Their stilt roots trap sediment, protect coastlines from storm surge, and "
    "shelter juvenile fish, which makes them one of the most productive habitats "
    "per hectare anywhere on the planet.",
    "The printing press spread through Europe in the second half of the "
    "fifteenth century. Within fifty years the cost of a book had fallen by "
    "more than an order of magnitude, and vernacular texts began to outnumber "
    "Latin ones in the catalogues of the larger printing houses.",
    "Superconductivity appears below a critical temperature at which electrons "
    "pair up and move through the lattice without resistance. Type II "
    "superconductors admit magnetic flux in quantised vortices, and pinning "
    "those vortices is what lets a magnet levitate stably above the material.",
    "The Silk Road was never a single road. It was a shifting network of caravan "
    "routes across Central Asia along which silk, paper, glassware and religious "
    "ideas moved in both directions, with most goods changing hands many times "
    "between the workshop and the buyer.",
    "Bees navigate using polarised skylight and a memory of landmarks near the "
    "hive. A returning forager communicates the direction and distance of a "
    "food source with a waggle dance whose angle encodes the bearing relative "
    "to the sun and whose duration encodes the distance.",
    "Double-entry bookkeeping was described in print by Luca Pacioli in 1494, "
    "though Venetian merchants had used it for generations. Every transaction "
    "is recorded twice, as a debit and a credit, so that an arithmetic check on "
    "the ledger catches most clerical errors before they compound.",
    "Plate tectonics explains why earthquakes cluster along narrow belts. Oceanic "
    "crust is created at spreading ridges and consumed at subduction zones, and "
    "the great majority of the planet's seismic energy is released where one "
    "plate grinds past or beneath another.",
    "The tuning of a piano is a compromise. Pure intervals cannot all be "
    "satisfied at once on a fixed-pitch instrument, so equal temperament spreads "
    "the discrepancy evenly across the twelve semitones, leaving every key "
    "slightly out of tune but all of them equally usable.",
    "Antibiotic resistance spreads faster than most people expect because "
    "bacteria exchange plasmids directly rather than only inheriting genes. A "
    "resistance cassette that arose in one species can appear in an unrelated "
    "one within a single hospital ward.",
    "Medieval cathedral builders worked without structural calculations. They "
    "relied on proportion rules handed down through masons' lodges, and on the "
    "evidence of what had stood elsewhere, which is why a collapse at Beauvais "
    "changed practice across a whole region.",
    "Photographic film records an image as a latent pattern of silver halide "
    "crystals that development amplifies. The grain that gives film its "
    "characteristic texture is the physical size of those crystals, so faster "
    "film is grainier for reasons that have nothing to do with the lens.",
    "The Dutch East India Company issued the first widely traded shares and, "
    "with them, the first market in speculative rumour. Amsterdam brokers were "
    "trading options and short positions within decades, long before anyone had "
    "a theory of what such contracts were worth.",
    "Sleep consolidates memory in stages. Slow-wave sleep appears to strengthen "
    "declarative traces in the hippocampus, while REM sleep is associated with "
    "procedural and emotional material, which is one reason total sleep "
    "deprivation degrades learning more than a shifted schedule does.",
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--out", required=True)
    p.add_argument("--av-out", default=None,
                   help="Also write an AV-SFT parquet here (prompt/response/\nactivation), so the smoke test can train a tiny AV that actually emits\n<explanation> tags instead of an instruct model's thinking block.")
    p.add_argument("--positions-per-doc", type=int, default=4)
    p.add_argument("--min-position", type=int, default=12)
    p.add_argument("--repeat-docs", type=int, default=1,
                   help="Repeat the document list this many times (with different "
                        "doc ids) so every split has rows to work with.")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nla.utils.arch_adapters import resolve_decoder_layers

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager",
    ).to(args.device).eval()
    d_model = model.config.hidden_size

    actor_template = (
        "You are a meticulous AI researcher conducting an important investigation "
        "into activation vectors from a language model. Your overall task is to "
        "describe the semantic content of that activation vector.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context. You "
        "must then produce an explanation for the vector, enclosed within "
        "<explanation> tags. The explanation consists of 2-3 text snippets "
        "describing that vector.\n\nHere is the vector:\n\n"
        "<concept>{injection_char}</concept>\n\nPlease provide an explanation."
    )
    # The sidecar keeps the {injection_char} slot; the parquet prompt column
    # carries the <INJECT> placeholder, exactly as datagen's stage 3 writes it.
    actor_prompt_content = actor_template.format(injection_char=INJECT_PLACEHOLDER)
    critic_template = "Summary of the following text: <text>{explanation}</text> <summary>"
    token_meta = build_token_meta(tok, actor_template, critic_template)

    captured = {}

    def hook(_m, _a, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    resolve_decoder_layers(model)[args.layer].register_forward_hook(hook)

    rows = []
    for rep in range(args.repeat_docs):
        for di, doc in enumerate(_DOCS):
            ids = tok(doc, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=args.device))
            h = captured["h"][0]                       # [T, d]
            n = len(ids)
            step = max(1, (n - args.min_position) // args.positions_per_doc)
            for k in range(args.positions_per_doc):
                pos = args.min_position + k * step
                if pos >= n - 1:
                    break
                prefix = tok.decode(ids[: pos + 1])
                topic = " ".join(doc.split()[:7])
                local = " ".join(prefix.split()[-8:])
                gold = wrap_explanation(
                    f"Expository prose about {topic.rstrip('.,')}.\n"
                    f"The text immediately before this point ends with "
                    f"\u201c{local}\u201d.\n"
                    f"What follows continues that sentence in the same factual register."
                )
                rows.append({
                    "response": gold,
                    "prompt": [{"role": "user", "content": actor_prompt_content}],
                    "activation_vector": h[pos].float().tolist(),
                    "n_raw_tokens": pos + 1,
                    "activation_layer": args.layer,
                    "doc_id": f"smoke-finefineweb:train:{rep * len(_DOCS) + di}",
                    "detokenized_text_truncated": prefix,
                })

    schema = pa.schema([
        ("prompt", pa.large_list(pa.struct([("role", pa.string()),
                                            ("content", pa.large_string())]))),
        ("activation_vector", pa.list_(pa.float32(), d_model)),
        ("n_raw_tokens", pa.int64()),
        ("activation_layer", pa.int64()),
        ("doc_id", pa.large_string()),
        ("detokenized_text_truncated", pa.large_string()),
    ])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([{k: v for k, v in r.items() if k != "response"} for r in rows],
                             schema=schema), str(out))

    meta = NLADatasetMeta(
        dataset_id=f"pred_smoke_{Path(args.model).name}_L{args.layer}",
        stage="rl", row_count=len(rows),
        extraction=NLAExtractionMeta(
            base_model=args.model, d_model=d_model, layer_index=args.layer,
            norm="none", corpus="pred-smoke-inline",
            corpus_slice={"start": 0, "length": len(_DOCS) * args.repeat_docs},
            positions_per_doc=args.positions_per_doc),
        tokens=token_meta,
        prompt_templates={"actor": actor_template, "critic": critic_template},
        created_by="scripts/make_pred_smoke_data.py",
    )
    sidecar_path_for(str(out)).write_text(serialize_sidecar(meta))

    if args.av_out:
        av_schema = pa.schema([schema.field(i) for i in range(len(schema))]
                              + [pa.field("response", pa.large_string())])
        av_out = Path(args.av_out)
        av_out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=av_schema), str(av_out))
        import dataclasses as _dc
        av_meta = _dc.replace(meta, stage="av_sft",
                              dataset_id=meta.dataset_id + "_av", created_at="")
        sidecar_path_for(str(av_out)).write_text(serialize_sidecar(av_meta))
        print(f"[smoke-data] AV-SFT parquet -> {av_out}")

    print(f"[smoke-data] {len(rows)} rows, d_model={d_model}, "
          f"marker id {token_meta.injection_token_id} -> {out}")


if __name__ == "__main__":
    main()
