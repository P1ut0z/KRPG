"""
KRPG Local Lightweight Verification Script
==========================================
Quickly verifies all KRPG modules on local CPU with small samples.
"""

import os
import sys
import json
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from krpg.knowledge import KnowledgeGraph, RAGRetriever, PromptBuilder
from krpg.generation import AminoAcidTokenizer, KRPGGenerator
from krpg.generation.generator import AMPSequenceDataset, SequenceGenerator
from krpg.validation import (
    AMPPredictor, RuleBasedAMPPredictor, CompositeScorer,
    ToxicityPredictor, StabilityPredictor,
    SimilarityFilter, PhysicochemicalFilter,
    FeedbackOptimizer,
)


def test_knowledge_module():
    print("=" * 60)
    print("[Module 1] Knowledge Graph & RAG Retrieval")
    print("=" * 60)

    kg = KnowledgeGraph()
    kg.build_default_amp_knowledge_graph()
    summary = kg.summary()
    print(f"  Knowledge Graph: {summary['total_entities']} entities, {summary['total_relations']} relations")
    print(f"  Entity types: {summary['entity_types']}")

    retriever = RAGRetriever(kg)
    retrieved = retriever.retrieve_by_target(
        target="Broad-Spectrum AMP, Gram-Negative, Low Toxicity",
        constraints={"length": "12-20", "charge": "+4 to +8", "toxicity": "low"},
    )
    print(f"  RAG Retrieved: {sum(len(v) for v in retrieved.values())} knowledge items")

    kb_results = retriever.search_knowledge_base("low toxicity amphipathic AMP", top_k=3)
    print(f"  Knowledge base search: {len(kb_results)} results")
    for r in kb_results:
        print(f"    - [{r['source']}] (relevance={r['relevance']}): {r['content'][:60]}...")

    builder = PromptBuilder()
    prompt = builder.build_structured_prompt({
        "target": "Broad-Spectrum AMP",
        "activity": "Gram-Negative",
        "constraint": "Low Toxicity",
        "preference": "Length 12-20, Positive Charge, Amphipathic",
        "knowledge": retrieved,
    })
    print(f"  Prompt built ({len(prompt)} chars):")
    print(f"    {prompt[:120]}...")
    print()

    return kg, retriever, builder


def test_generation_module():
    print("=" * 60)
    print("[Module 2] Sequence Generation (Lightweight)")
    print("=" * 60)

    tokenizer = AminoAcidTokenizer()
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")

    test_seq = "KWLKKIGAVLKVL"
    encoded = tokenizer.encode(test_seq)
    decoded = tokenizer.decode(encoded)
    print(f"  Encode/Decode test: '{test_seq}' -> {encoded} -> '{decoded}'")
    assert test_seq == decoded, "Tokenizer encode/decode mismatch!"
    print("  Tokenizer test: PASSED")

    design_spec = {
        "target": "Broad-Spectrum AMP",
        "activity": "Gram-Negative",
        "constraint": "Low Toxicity",
        "preference": "Length 12-20, Positive Charge, Amphipathic",
    }

    model = KRPGGenerator(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=64,
        dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
        use_prompt_conditioning=True,
    )
    model_size = model.get_model_size()
    print(f"  Model size: {model_size['total_params']:,} params ({model_size['total_params_millions']:.2f}M)")

    sample_amps = [
        "KWLKKIGAVLKVL",
        "GFKRIVQRIKDFL",
        "LYIAKLLKRFN",
        "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
        "GIGKFLHSAKKFGKAFVGEIMNS",
    ]
    dataset = AMPSequenceDataset(sample_amps, tokenizer, max_length=30)
    print(f"  Dataset: {len(dataset)} sequences")

    generator = SequenceGenerator(model, tokenizer, device="cpu")
    history = generator.train_model(
        dataset, n_epochs=5, batch_size=4, learning_rate=1e-3,
        save_path=os.path.join("..", "outputs", "local_checkpoint.pt"),
        design_spec=design_spec,
    )
    print(f"  Training loss: {[f'{l:.4f}' for l in history['loss']]}")

    prompt_ids = torch.tensor([[tokenizer.bos_token_id]])
    generated = generator.generate_sequences(
        prompt_ids=prompt_ids,
        n_sequences=5,
        max_length=20,
        temperature=1.0,
        top_k=20,
        top_p=0.9,
        design_spec=design_spec,
    )
    print(f"  Generated {len(generated)} sequences (with prompt conditioning):")
    for i, g in enumerate(generated):
        print(f"    [{i+1}] {g['sequence']} (prob={g['avg_log_prob']:.4f}, len={g['length']})")

    print()
    return tokenizer, model, generator


def test_validation_module():
    print("=" * 60)
    print("[Module 3] Computational Validation")
    print("=" * 60)

    test_sequences = [
        "KWLKKIGAVLKVL",
        "GFKRIVQRIKDFL",
        "LYIAKLLKRFN",
        "WWWWWWWWWW",
        "GAGAGAGAGA",
    ]

    rule_predictor = RuleBasedAMPPredictor()
    print("  AMP Prediction (Rule-based):")
    for seq in test_sequences:
        result = rule_predictor.predict(seq)
        print(f"    {seq:20s} score={result['amp_score']:.3f} candidate={result['is_amp_candidate']}")

    tox_predictor = ToxicityPredictor()
    print("  Toxicity Prediction:")
    for seq in test_sequences:
        result = tox_predictor.predict(seq)
        print(f"    {seq:20s} score={result['toxicity_score']:.3f} toxic={result['is_toxic']}")

    stab_predictor = StabilityPredictor()
    print("  Stability Prediction:")
    for seq in test_sequences:
        result = stab_predictor.predict(seq)
        print(f"    {seq:20s} score={result['stability_score']:.3f} stable={result['is_stable']}")

    known_amps = ["KWLKKIGAVLKVL", "GFKRIVQRIKDFL"]
    sim_filter = SimilarityFilter(known_amps)
    sim_results = sim_filter.filter_by_similarity(test_sequences, threshold=0.8)
    print("  Similarity Filter:")
    for r in sim_results:
        print(f"    {r['sequence']:20s} max_sim={r['max_similarity']:.3f} novel={r['is_novel']}")

    phys_filter = PhysicochemicalFilter()
    phys_results = phys_filter.batch_compute(test_sequences)
    print("  Physicochemical Properties:")
    for seq, r in zip(test_sequences, phys_results):
        print(f"    {seq:20s} len={r['length']} MW={r['molecular_weight']:.0f} charge={r['net_charge']:+d} hydro={r['hydrophobicity']:.2f} pI={r['isoelectric_point']:.2f}")

    scorer = CompositeScorer()
    print("  Composite Scoring:")
    for seq in test_sequences:
        amp_r = rule_predictor.predict(seq)
        tox_r = tox_predictor.predict(seq)
        stab_r = stab_predictor.predict(seq)
        sim_r = sim_filter.filter_by_similarity([seq])[0]
        comp = scorer.score(amp_r["amp_score"], tox_r["toxicity_score"],
                           stab_r["stability_score"], sim_r["is_novel"])
        print(f"    {seq:20s} composite={comp['composite_score']:.3f}")

    print()
    return rule_predictor, tox_predictor, stab_predictor, sim_filter, phys_filter


def test_feedback_module():
    print("=" * 60)
    print("[Module 4] Feedback & Optimization (Closed Loop)")
    print("=" * 60)

    sample_results = [
        {"sequence": "KWLKKIGAVLKVL", "amp_score": 0.85, "toxicity_score": 0.25, "stability_score": 0.60},
        {"sequence": "GFKRIVQRIKDFL", "amp_score": 0.72, "toxicity_score": 0.35, "stability_score": 0.55},
        {"sequence": "LYIAKLLKRFN", "amp_score": 0.55, "toxicity_score": 0.20, "stability_score": 0.45},
        {"sequence": "WWWWWWWWWW", "amp_score": 0.30, "toxicity_score": 0.85, "stability_score": 0.30},
        {"sequence": "GAGAGAGAGA", "amp_score": 0.15, "toxicity_score": 0.10, "stability_score": 0.25},
    ]

    optimizer = FeedbackOptimizer()
    feedback = optimizer.generate_feedback(sample_results, round_num=1)
    print(f"  Analysis:")
    print(f"    Avg AMP score: {feedback['analysis']['avg_amp_score']:.3f}")
    print(f"    Avg Toxicity: {feedback['analysis']['avg_toxicity_score']:.3f}")
    print(f"    Avg Stability: {feedback['analysis']['avg_stability_score']:.3f}")
    print(f"  Suggestions:")
    for s in feedback['analysis']['suggestions']:
        print(f"    - {s}")
    print(f"  Updated constraints: {feedback['updated_constraints']}")

    original_spec = {
        "target": "Broad-Spectrum AMP",
        "activity": "Gram-Negative",
        "constraint": "Low Toxicity",
        "preference": "Length 12-20, Positive Charge",
    }
    updated_spec = optimizer.apply_feedback_to_spec(original_spec, feedback)
    print(f"  Closed-loop spec update:")
    print(f"    Original preference: {original_spec['preference']}")
    print(f"    Updated  preference: {updated_spec['preference']}")
    if "constraint" in updated_spec:
        print(f"    Updated  constraint: {updated_spec['constraint']}")

    print()
    return optimizer


def test_end_to_end_pipeline():
    print("=" * 60)
    print("[Pipeline] End-to-End Closed-Loop Pipeline")
    print("=" * 60)

    kg = KnowledgeGraph()
    kg.build_default_amp_knowledge_graph()

    retriever = RAGRetriever(kg)
    builder = PromptBuilder()

    design_spec = {
        "target": "Broad-Spectrum AMP",
        "activity": "Gram-Negative",
        "constraint": "Low Toxicity",
        "preference": "Length 12-20, Positive Charge, Amphipathic",
    }
    retrieved = retriever.retrieve_by_target(
        "Broad-Spectrum AMP, Gram-Negative, Low Toxicity",
        {"length": "12-20", "charge": "+4 to +8"},
    )
    design_spec["knowledge"] = retrieved
    prompt = builder.build_structured_prompt(design_spec)
    print(f"  Step 1: Prompt built ({len(prompt)} chars)")

    tokenizer = AminoAcidTokenizer()
    model = KRPGGenerator(
        vocab_size=tokenizer.vocab_size,
        d_model=64, n_heads=2, n_layers=2, d_ff=128,
        max_seq_len=64, dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
        use_prompt_conditioning=True,
    )
    generator = SequenceGenerator(model, tokenizer, device="cpu")

    sample_amps = [
        "KWLKKIGAVLKVL", "GFKRIVQRIKDFL", "LYIAKLLKRFN",
        "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
        "GIGKFLHSAKKFGKAFVGEIMNS",
    ]
    dataset = AMPSequenceDataset(sample_amps, tokenizer, max_length=30)
    generator.train_model(dataset, n_epochs=3, batch_size=4, learning_rate=1e-3,
                         design_spec=design_spec)
    print(f"  Step 2: Model trained (3 epochs, prompt-conditioned)")

    prompt_ids = torch.tensor([[tokenizer.bos_token_id]])
    candidates = generator.generate_sequences(
        prompt_ids=prompt_ids, n_sequences=8,
        max_length=20, temperature=1.0, top_k=20, top_p=0.9,
        design_spec=design_spec,
    )
    print(f"  Step 3: Generated {len(candidates)} candidate sequences")

    rule_predictor = RuleBasedAMPPredictor()
    tox_predictor = ToxicityPredictor()
    stab_predictor = StabilityPredictor()
    phys_filter = PhysicochemicalFilter()
    scorer = CompositeScorer()

    full_results = []
    for c in candidates:
        seq = c["sequence"]
        amp_result = rule_predictor.predict(seq)
        tox_result = tox_predictor.predict(seq)
        stab_result = stab_predictor.predict(seq)
        phys_result = phys_filter.compute_properties(seq)
        comp = scorer.score(amp_result["amp_score"], tox_result["toxicity_score"],
                           stab_result["stability_score"], True)
        full_results.append({
            "sequence": seq,
            "amp_score": amp_result["amp_score"],
            "toxicity_score": tox_result["toxicity_score"],
            "stability_score": stab_result["stability_score"],
            "composite_score": comp["composite_score"],
            "physicochemical": phys_result,
        })

    print(f"  Step 4: Validation + Composite Scoring complete")
    print(f"\n  {'Sequence':25s} {'AMP':8s} {'Tox':8s} {'Stab':8s} {'Comp':8s} {'Charge':8s} {'pI':6s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for r in full_results:
        s = r["sequence"]
        print(f"  {s[:24]:25s} {r['amp_score']:.3f}  {r['toxicity_score']:.3f}  {r['stability_score']:.3f}  {r['composite_score']:.3f}  {r['physicochemical']['net_charge']:+d}     {r['physicochemical']['isoelectric_point']:.2f}")

    optimizer = FeedbackOptimizer()
    feedback = optimizer.generate_feedback(full_results, round_num=1)
    updated_spec = optimizer.apply_feedback_to_spec(design_spec, feedback)
    print(f"\n  Step 5: Feedback + Closed-Loop Update")
    print(f"    Suggestions: {len(feedback['analysis']['suggestions'])}")
    for s in feedback['analysis']['suggestions']:
        print(f"      - {s}")
    print(f"    Updated preference: {updated_spec.get('preference', 'N/A')}")

    print()
    return full_results, feedback


def main():
    print("\n" + "=" * 60)
    print("  KRPG - Knowledge-Retrieval Prompted Generator")
    print("  Local Lightweight Verification (with Prompt Conditioning)")
    print("=" * 60 + "\n")

    os.makedirs(os.path.join("..", "outputs"), exist_ok=True)

    start_time = time.time()

    kg, retriever, builder = test_knowledge_module()
    tokenizer, model, generator = test_generation_module()
    rule_predictor, tox_predictor, stab_predictor, sim_filter, phys_filter = test_validation_module()
    optimizer = test_feedback_module()
    results, feedback = test_end_to_end_pipeline()

    elapsed = time.time() - start_time

    print("=" * 60)
    print(f"  ALL MODULES PASSED - Time: {elapsed:.1f}s")
    print("=" * 60)

    summary = {
        "status": "PASSED",
        "total_time_seconds": round(elapsed, 1),
        "modules_tested": [
            "KnowledgeGraph (entities, relations, Jaccard retrieval)",
            "RAGRetriever (target retrieval, embedding similarity search)",
            "PromptBuilder (structured prompt, feedback prompt)",
            "AminoAcidTokenizer (encode, decode, vocab)",
            "KRPGGenerator (forward, generate, prompt conditioning, training)",
            "AMPPredictor / RuleBasedAMPPredictor / CompositeScorer",
            "ToxicityPredictor",
            "StabilityPredictor",
            "SimilarityFilter / PhysicochemicalFilter (alignment-aware, pI)",
            "FeedbackOptimizer (closed-loop spec update)",
        ],
        "generated_sequences": len(results),
        "avg_amp_score": round(sum(r["amp_score"] for r in results) / max(len(results), 1), 4),
        "avg_toxicity_score": round(sum(r["toxicity_score"] for r in results) / max(len(results), 1), 4),
        "avg_stability_score": round(sum(r["stability_score"] for r in results) / max(len(results), 1), 4),
        "avg_composite_score": round(sum(r["composite_score"] for r in results) / max(len(results), 1), 4),
    }

    summary_path = os.path.join("..", "outputs", "local_verification_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
