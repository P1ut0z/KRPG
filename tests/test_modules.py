"""
KRPG Module Unit Tests
Run: python -m pytest tests/test_modules.py -v
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from krpg.knowledge import KnowledgeGraph, RAGRetriever, PromptBuilder, PromptParser
from krpg.generation import AminoAcidTokenizer, KRPGGenerator
from krpg.generation.generator import AMPSequenceDataset, SequenceGenerator
from krpg.generation.model import PromptEncoder
from krpg.validation import (
    RuleBasedAMPPredictor, CompositeScorer,
    ToxicityPredictor, StabilityPredictor,
    SimilarityFilter, PhysicochemicalFilter,
    FeedbackOptimizer,
)


def test_knowledge_graph():
    kg = KnowledgeGraph()
    kg.build_default_amp_knowledge_graph()
    added = kg.build_from_amp_records([
        {"sequence": "KWLKKIGAVLKVL", "label": 1, "source": "unit_test"},
        {"sequence": "GFKRIVQRIKDFL", "label": 1, "source": "unit_test"},
    ])
    assert added == 2
    summary = kg.summary()
    assert summary["total_entities"] > 0
    assert summary["total_relations"] > 0
    assert "property" in summary["entity_types"]
    assert "peptide" in summary["entity_types"]

    neighbors = kg.get_neighbors("positive_charge")
    assert len(neighbors["outgoing"]) > 0 or len(neighbors["incoming"]) > 0

    save_path = os.path.join(os.path.dirname(__file__), "test_kg.json")
    kg.save(save_path)
    kg2 = KnowledgeGraph()
    kg2.load(save_path)
    assert len(kg2.entities) == len(kg.entities)
    os.remove(save_path)


def test_rag_retriever():
    kg = KnowledgeGraph()
    kg.build_default_amp_knowledge_graph()
    kg.build_from_amp_records([
        {"sequence": "KWLKKIGAVLKVL", "label": 1, "source": "unit_test"},
        {"sequence": "LYIAKLLKRFN", "label": 1, "source": "unit_test"},
    ])
    retriever = RAGRetriever(kg)

    retrieved = retriever.retrieve_by_target("Broad-Spectrum AMP, Low Toxicity")
    assert isinstance(retrieved, dict)
    total_items = sum(len(v) for v in retrieved.values())
    assert total_items > 0

    results = retriever.search_knowledge_base("low toxicity AMP", top_k=3)
    assert len(results) <= 3
    assert all("relevance" in r for r in results)

    context = retriever.retrieve_by_design_spec({
        "target": "Broad-Spectrum AMP",
        "activity": "Gram-Negative",
        "constraint": "Low Toxicity",
        "preference": "Length 12-20, Positive Charge, Amphipathic",
    })
    assert "property_constraints" in context
    assert "activity_evidence" in context
    assert len(context["property_constraints"]) > 0


def test_prompt_builder():
    builder = PromptBuilder()
    prompt = builder.build_prompt(
        target="Broad-Spectrum AMP",
        activity="Gram-Negative",
        constraint="Low Toxicity",
        preference="Length 12-20, Positive Charge",
    )
    assert "Target:" in prompt
    assert "Broad-Spectrum AMP" in prompt

    prompt2 = builder.build_structured_prompt({
        "target": "Test AMP",
        "activity": "Test",
        "constraint": "Test",
        "preference": "Test",
    })
    assert "Test AMP" in prompt2

    parser = PromptParser()
    spec = parser.parse("生成广谱、低毒、长度12-20、正电荷、两亲性的AMP")
    record = builder.build_prompt_record(
        raw_user_prompt=spec["raw_prompt"],
        design_spec=spec,
        retrieved_context={"property_constraints": [{"text": "Length 12-20", "source": "test"}]},
    )
    assert "rendered_prompt" in record
    assert "Retrieved Knowledge" in record["rendered_prompt"]


def test_tokenizer():
    tokenizer = AminoAcidTokenizer()
    assert tokenizer.vocab_size == 25
    assert tokenizer.pad_token_id == 0
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2

    seq = "KWLKKIGAVLKVL"
    encoded = tokenizer.encode(seq)
    decoded = tokenizer.decode(encoded)
    assert decoded == seq

    batch = tokenizer.batch_encode(["KWL", "GAV"], max_length=10)
    assert len(batch) == 2
    assert len(batch[0]) == 10


def test_generator_model():
    tokenizer = AminoAcidTokenizer()
    model = KRPGGenerator(
        vocab_size=tokenizer.vocab_size,
        d_model=32, n_heads=2, n_layers=2, d_ff=64,
        max_seq_len=32, dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
        use_prompt_conditioning=True,
    )
    size_info = model.get_model_size()
    assert size_info["total_params"] > 0

    batch = torch.randint(0, tokenizer.vocab_size, (2, 10))
    logits, loss = model(input_ids=batch, labels=batch)
    assert logits.shape == (2, 10, tokenizer.vocab_size)
    assert loss is not None and loss.item() > 0


def test_prompt_encoder():
    encoder = PromptEncoder(d_model=64)
    spec = {
        "target": "Broad-Spectrum AMP",
        "constraint": "Low Toxicity",
        "preference": "Length 12-20, Positive Charge",
    }
    features = encoder.encode_spec(spec)
    assert features.shape == (1, encoder.PROPERTY_DIM)
    output = encoder(features)
    assert output.shape == (1, 64)


def test_prompt_conditioning():
    tokenizer = AminoAcidTokenizer()
    model = KRPGGenerator(
        vocab_size=tokenizer.vocab_size,
        d_model=32, n_heads=2, n_layers=2, d_ff=64,
        max_seq_len=32, dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
        use_prompt_conditioning=True,
    )

    spec = {"target": "Broad-Spectrum AMP", "constraint": "Low Toxicity",
            "preference": "Length 12-20, Positive Charge",
            "retrieved_context": {"property_constraints": [
                {"feature": "length", "range": [12, 20], "text": "Length range"},
                {"feature": "net_charge", "range": [4, 8], "text": "Charge range"},
            ], "motif_hints": [{"text": "amphipathic helix"}]}}
    prompt_vector = model.encode_prompt(spec)
    assert prompt_vector is not None
    assert prompt_vector.shape[0] == 1

    batch = torch.randint(0, tokenizer.vocab_size, (2, 10))
    pv = prompt_vector.expand(2, -1)
    logits, loss = model(input_ids=batch, labels=batch, prompt_vector=pv)
    assert logits.shape == (2, 10, tokenizer.vocab_size)

    prompt_ids = torch.tensor([[tokenizer.bos_token_id]])
    generated = model.generate(prompt_ids, max_new_tokens=10, prompt_vector=prompt_vector[:1])
    assert generated.shape[1] > 1


def test_sequence_generator():
    tokenizer = AminoAcidTokenizer()
    model = KRPGGenerator(
        vocab_size=tokenizer.vocab_size,
        d_model=32, n_heads=2, n_layers=2, d_ff=64,
        max_seq_len=32, dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
        use_prompt_conditioning=True,
    )
    generator = SequenceGenerator(model, tokenizer, device="cpu")

    dataset = AMPSequenceDataset(["KWLKKIGAVLKVL", "GFKRIVQRIKDFL"], tokenizer, max_length=20)
    assert len(dataset) == 2

    design_spec = {"target": "Broad-Spectrum AMP", "constraint": "Low Toxicity",
                   "preference": "Length 12-20, Positive Charge"}
    history = generator.train_model(dataset, n_epochs=2, batch_size=2, learning_rate=1e-3,
                                   design_spec=design_spec)
    assert "loss" in history
    assert len(history["loss"]) == 2

    prompt_ids = torch.tensor([[tokenizer.bos_token_id]])
    prompt_record = {
        "design_spec": design_spec,
        "retrieved_context": {"property_constraints": [{"text": "Length 12-20"}]},
        "rendered_prompt": "Target: Broad-Spectrum AMP",
    }
    results = generator.generate_sequences(prompt_ids, n_sequences=3, max_length=10,
                                          prompt_record=prompt_record)
    assert len(results) == 3
    for r in results:
        assert "sequence" in r
        assert "prompt" in r
        assert "retrieved_knowledge" in r


def test_amp_predictor():
    predictor = RuleBasedAMPPredictor()
    result = predictor.predict("KWLKKIGAVLKVL")
    assert "amp_score" in result
    assert "is_amp_candidate" in result
    assert 0 <= result["amp_score"] <= 1

    result_short = predictor.predict("KW")
    assert result_short["amp_score"] == 0.0


def test_composite_scorer():
    scorer = CompositeScorer()
    result = scorer.score(amp_score=0.8, toxicity_score=0.2, stability_score=0.6, is_novel=True)
    assert "composite_score" in result
    assert "breakdown" in result
    assert result["composite_score"] > 0
    assert result["composite_score"] <= 1.0

    result_not_novel = scorer.score(amp_score=0.8, toxicity_score=0.2, stability_score=0.6, is_novel=False)
    assert result_not_novel["composite_score"] < result["composite_score"]


def test_toxicity_predictor():
    predictor = ToxicityPredictor()
    result = predictor.predict("KWLKKIGAVLKVL")
    assert "toxicity_score" in result
    assert "is_toxic" in result
    assert 0 <= result["toxicity_score"] <= 1


def test_stability_predictor():
    predictor = StabilityPredictor()
    result = predictor.predict("KWLKKIGAVLKVL")
    assert "stability_score" in result
    assert "is_stable" in result
    assert 0 <= result["stability_score"] <= 1


def test_similarity_filter():
    filter_ = SimilarityFilter(known_amps=["KWLKKIGAVLKVL"])
    results = filter_.filter_by_similarity(["KWLKKIGAVLKVL", "AAAAAAA"], threshold=0.8)
    assert results[0]["is_novel"] == False
    assert results[1]["is_novel"] == True

    dedup = filter_.deduplicate(["AAAA", "AAAA", "BBBB"], identity_threshold=0.9)
    assert len(dedup) == 2

    sim_diff_len = filter_.sequence_identity("KWLKK", "KWLKKIGAVLKVL")
    assert sim_diff_len > 0.0


def test_physicochemical_filter():
    filter_ = PhysicochemicalFilter()
    props = filter_.compute_properties("KWLKKIGAVLKVL")
    assert props["length"] == 13
    assert props["net_charge"] > 0
    assert props["molecular_weight"] > 0
    assert props["isoelectric_point"] > 0.0
    assert props["hydrophobic_ratio"] > 0.0
    assert props["aromatic_ratio"] >= 0.0


def test_feedback_optimizer():
    optimizer = FeedbackOptimizer()
    results = [
        {"sequence": "KWLKKIGAVLKVL", "amp_score": 0.85, "toxicity_score": 0.25, "stability_score": 0.60},
        {"sequence": "GFKRIVQRIKDFL", "amp_score": 0.72, "toxicity_score": 0.35, "stability_score": 0.55},
    ]
    feedback = optimizer.generate_feedback(results, round_num=1)
    assert "analysis" in feedback
    assert "updated_constraints" in feedback
    assert feedback["round"] == 1

    original_spec = {
        "target": "Broad-Spectrum AMP",
        "constraint": "Low Toxicity",
        "preference": "Length 12-20",
    }
    updated_spec = optimizer.apply_feedback_to_spec(original_spec, feedback)
    assert "preference" in updated_spec


def test_model_backward_compat():
    """Test that model works without prompt conditioning (backward compat)."""
    tokenizer = AminoAcidTokenizer()
    model = KRPGGenerator(
        vocab_size=tokenizer.vocab_size,
        d_model=32, n_heads=2, n_layers=2, d_ff=64,
        max_seq_len=32, dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
        use_prompt_conditioning=False,
    )
    batch = torch.randint(0, tokenizer.vocab_size, (2, 10))
    logits, loss = model(input_ids=batch, labels=batch)
    assert logits.shape == (2, 10, tokenizer.vocab_size)
    assert loss is not None
