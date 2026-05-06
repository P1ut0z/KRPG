"""
KRPG Server-Side Full Deployment Script
========================================
Full deployment with GPU, large model, complete training, and closed-loop optimization.

Usage:
  conda activate KRPG
  python scripts/run_server.py --mode train        # Train generation model
  python scripts/run_server.py --mode generate     # Generate candidate sequences
  python scripts/run_server.py --mode evaluate     # Evaluate candidate sequences
  python scripts/run_server.py --mode full         # Full pipeline (single round)
  python scripts/run_server.py --mode pipeline     # Multi-round closed-loop optimization
"""

import os
import sys
import json
import csv
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from krpg.knowledge import KnowledgeGraph, RAGRetriever, PromptBuilder
from krpg.generation import AminoAcidTokenizer, KRPGGenerator
from krpg.generation.generator import AMPSequenceDataset, SequenceGenerator
from krpg.validation import (
    RuleBasedAMPPredictor, AMPPredictor, CompositeScorer,
    ToxicityPredictor, StabilityPredictor,
    SimilarityFilter, PhysicochemicalFilter,
    FeedbackOptimizer,
)


def setup_logging(log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"server_run_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA not available, using CPU (training will be slow)")
    return device


def load_amp_data(data_path: str = None) -> List[str]:
    if data_path is None:
        default_path = os.path.join("data", "amp_sequences.json")
        if os.path.exists(default_path):
            data_path = default_path

    if data_path and os.path.exists(data_path):
        ext = os.path.splitext(data_path)[1].lower()
        if ext == ".csv":
            with open(data_path, "r", encoding="utf-8-sig", newline="") as f:
                sequences = [
                    row["sequence"].strip().upper()
                    for row in csv.DictReader(f)
                    if row.get("sequence")
                ]
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sequences = data if isinstance(data, list) else [item["sequence"] for item in data]
            sequences = [item["sequence"] if isinstance(item, dict) else item for item in sequences]
        logging.info(f"Loaded {len(sequences)} AMP sequences from {data_path}")
        return sequences

    default_amps = [
        "KWLKKIGAVLKVL", "GFKRIVQRIKDFL", "LYIAKLLKRFN",
        "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
        "GIGKFLHSAKKFGKAFVGEIMNS",
        "RWKWRRWW", "KRFKKFFKKLK", "FLPIIAKDLL", "GLLSALKKLL",
        "KIAKVALKAL", "KWLKKIGAVLKVLTTG", "GFKRIVQRIKDFLRNLV",
        "LYIAKLLKRFNK", "KRFKKFFKKLKKL", "FLPIIAKDLLR",
        "GLLSALKKLLG", "KIAKVALKALKV", "RWKWRRWWR",
        "KRFKKFFKKLKKLK", "FLPIIAKDLLRG", "GLLSALKKLLGS",
        "KIAKVALKALKVA", "KWLKKIGAVLKVLT", "GFKRIVQRIKDFLRN",
    ]
    logging.info(f"Using {len(default_amps)} default AMP sequences")
    return default_amps


def build_knowledge_module():
    logging.info("[KG] Building knowledge graph...")
    kg = KnowledgeGraph()
    kg_dir = os.path.join("data", "knowledge_base")
    if os.path.exists(os.path.join(kg_dir, "entities.jsonl")) and os.path.exists(os.path.join(kg_dir, "relations.jsonl")):
        kg.load_jsonl(kg_dir)
        logging.info(f"[KG] Loaded mature knowledge graph from {kg_dir}")
    else:
        kg.build_default_amp_knowledge_graph()
        logging.info("[KG] Mature graph not found; using default AMP design rules")
    summary = kg.summary()
    logging.info(f"[KG] Built: {summary['total_entities']} entities, {summary['total_relations']} relations")

    retriever = RAGRetriever(kg)
    builder = PromptBuilder()
    return kg, retriever, builder


def build_generation_model(tokenizer: AminoAcidTokenizer, device: torch.device,
                           d_model: int = 256, n_heads: int = 4,
                           n_layers: int = 4, d_ff: int = 512) -> KRPGGenerator:
    logging.info(f"[Model] Building KRPGGenerator: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
    model = KRPGGenerator(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=64,
        dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
        use_prompt_conditioning=True,
    )
    model_size = model.get_model_size()
    logging.info(f"[Model] Model size: {model_size['total_params']:,} params ({model_size['total_params_millions']:.2f}M)")
    return model


def evaluate_sequences(sequences: List[str], known_amps: List[str] = None,
                       scorer: CompositeScorer = None) -> List[Dict]:
    """Evaluate sequences with all predictors and composite scoring."""
    rule_predictor = RuleBasedAMPPredictor()
    tox_predictor = ToxicityPredictor()
    stab_predictor = StabilityPredictor()
    sim_filter = SimilarityFilter(known_amps or sequences[:10])
    phys_filter = PhysicochemicalFilter()
    if scorer is None:
        scorer = CompositeScorer()

    results = []
    for seq in sequences:
        amp_result = rule_predictor.predict(seq)
        tox_result = tox_predictor.predict(seq)
        stab_result = stab_predictor.predict(seq)
        phys_result = phys_filter.compute_properties(seq)
        sim_result = sim_filter.filter_by_similarity([seq])[0]

        comp = scorer.score(
            amp_result["amp_score"], tox_result["toxicity_score"],
            stab_result["stability_score"], sim_result["is_novel"],
        )

        results.append({
            "sequence": seq,
            "amp_score": amp_result["amp_score"],
            "toxicity_score": tox_result["toxicity_score"],
            "stability_score": stab_result["stability_score"],
            "is_novel": sim_result["is_novel"],
            "max_similarity": sim_result["max_similarity"],
            "physicochemical": phys_result,
            "composite_score": comp["composite_score"],
        })

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results


def train_mode(args, device: torch.device):
    logging.info("=" * 60)
    logging.info("MODE: TRAIN")
    logging.info("=" * 60)

    tokenizer = AminoAcidTokenizer()
    sequences = load_amp_data(args.data_path)

    design_spec = {
        "target": "Broad-Spectrum AMP",
        "activity": "Gram-Negative",
        "constraint": "Low Toxicity",
        "preference": "Length 12-20, Positive Charge, Amphipathic",
    }

    model = build_generation_model(tokenizer, device, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers)
    generator = SequenceGenerator(model, tokenizer, device=str(device))

    dataset = AMPSequenceDataset(sequences, tokenizer, max_length=args.max_length)
    logging.info(f"[Train] Dataset: {len(dataset)} sequences, max_length={args.max_length}")

    save_dir = args.save_dir or os.path.join("outputs", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"krpg_model_epoch{args.epochs}.pt")

    logging.info(f"[Train] Starting training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    start_time = time.time()
    history = generator.train_model(
        dataset,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=save_path,
        design_spec=design_spec,
    )
    elapsed = time.time() - start_time
    logging.info(f"[Train] Training complete: {elapsed:.1f}s")
    logging.info(f"[Train] Loss history: {[f'{l:.4f}' for l in history['loss']]}")
    logging.info(f"[Train] Model saved to: {save_path}")

    return generator


def generate_mode(args, device: torch.device):
    logging.info("=" * 60)
    logging.info("MODE: GENERATE")
    logging.info("=" * 60)

    tokenizer = AminoAcidTokenizer()
    model = build_generation_model(tokenizer, device, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers)

    generator = SequenceGenerator(model, tokenizer, str(device))

    if args.checkpoint and os.path.exists(args.checkpoint):
        generator.load_checkpoint(args.checkpoint)
        logging.info(f"[Generate] Loaded checkpoint: {args.checkpoint}")
    else:
        logging.warning("[Generate] No checkpoint loaded, using untrained model")

    kg, retriever, builder = build_knowledge_module()

    design_targets = [
        {"target": "Broad-Spectrum AMP", "activity": "Gram-Negative", "constraint": "Low Toxicity",
         "preference": "Length 12-20, Positive Charge, Amphipathic"},
        {"target": "Gram-Positive AMP", "activity": "Gram-Positive", "constraint": "Low Toxicity",
         "preference": "Length 15-25, Moderate Charge, Helical"},
        {"target": "Low Toxicity AMP", "activity": "Broad-Spectrum", "constraint": "Very Low Toxicity",
         "preference": "Length 10-18, Moderate Charge, Balanced Hydrophobicity"},
    ]

    all_candidates = []
    for i, spec in enumerate(design_targets):
        logging.info(f"[Generate] Target {i+1}: {spec['target']}")
        retrieved = retriever.retrieve_by_target(
            f"{spec['target']}, {spec['activity']}, {spec['constraint']}",
            {"length": spec['preference'].split(',')[0].replace('Length ', '')},
        )
        spec["knowledge"] = retrieved

        prompt_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
        candidates = generator.generate_sequences(
            prompt_ids=prompt_ids,
            n_sequences=args.n_sequences,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            design_spec=spec,
        )
        logging.info(f"[Generate] Generated {len(candidates)} sequences for target {i+1}")
        for j, c in enumerate(candidates[:3]):
            logging.info(f"  [{j+1}] {c['sequence']} (prob={c['avg_log_prob']:.4f})")

        all_candidates.extend(candidates)

    output_dir = args.output_dir or os.path.join("outputs", "generated")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2)
    logging.info(f"[Generate] All candidates saved to: {output_path}")

    return all_candidates


def evaluate_mode(args, device: torch.device):
    logging.info("=" * 60)
    logging.info("MODE: EVALUATE")
    logging.info("=" * 60)

    if args.input_file and os.path.exists(args.input_file):
        with open(args.input_file, "r") as f:
            candidates = json.load(f)
        sequences = [c["sequence"] for c in candidates]
        logging.info(f"[Evaluate] Loaded {len(sequences)} candidates from {args.input_file}")
    else:
        sequences = load_amp_data(args.data_path)
        logging.info(f"[Evaluate] Using {len(sequences)} AMP sequences for evaluation")

    results = evaluate_sequences(sequences)

    output_dir = args.output_dir or os.path.join("outputs", "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info(f"[Evaluate] Results saved to: {output_path}")
    logging.info(f"[Evaluate] Top 5 candidates by composite score:")
    for i, r in enumerate(results[:5]):
        logging.info(f"  [{i+1}] {r['sequence']:25s} composite={r['composite_score']:.3f} "
                     f"AMP={r['amp_score']:.3f} Tox={r['toxicity_score']:.3f} Stab={r['stability_score']:.3f}")

    avg_scores = {
        "avg_amp": sum(r["amp_score"] for r in results) / len(results),
        "avg_toxicity": sum(r["toxicity_score"] for r in results) / len(results),
        "avg_stability": sum(r["stability_score"] for r in results) / len(results),
        "avg_composite": sum(r["composite_score"] for r in results) / len(results),
        "n_novel": sum(1 for r in results if r["is_novel"]),
    }
    logging.info(f"[Evaluate] Averages: AMP={avg_scores['avg_amp']:.3f}, "
                 f"Tox={avg_scores['avg_toxicity']:.3f}, Stab={avg_scores['avg_stability']:.3f}, "
                 f"Composite={avg_scores['avg_composite']:.3f}, Novel={avg_scores['n_novel']}/{len(results)}")

    return results


def full_pipeline_mode(args, device: torch.device):
    logging.info("=" * 60)
    logging.info("MODE: FULL PIPELINE (with Prompt Conditioning)")
    logging.info("=" * 60)

    tokenizer = AminoAcidTokenizer()
    sequences = load_amp_data(args.data_path)

    design_spec = {
        "target": "Broad-Spectrum AMP",
        "activity": "Gram-Negative",
        "constraint": "Low Toxicity",
        "preference": "Length 12-20, Positive Charge, Amphipathic",
    }

    model = build_generation_model(tokenizer, device, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers)
    generator = SequenceGenerator(model, tokenizer, str(device))

    dataset = AMPSequenceDataset(sequences, tokenizer, max_length=args.max_length)
    save_dir = args.save_dir or os.path.join("outputs", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "krpg_model_full.pt")

    logging.info(f"[Pipeline] Phase 1: Training ({args.epochs} epochs, prompt-conditioned)")
    history = generator.train_model(dataset, n_epochs=args.epochs, batch_size=args.batch_size,
                                   learning_rate=args.lr, save_path=save_path,
                                   design_spec=design_spec)
    logging.info(f"[Pipeline] Training complete. Final loss: {history['loss'][-1]:.4f}")

    kg, retriever, builder = build_knowledge_module()
    retrieved = retriever.retrieve_by_target(
        "Broad-Spectrum AMP, Gram-Negative, Low Toxicity",
        {"length": "12-20", "charge": "+4 to +8"},
    )
    design_spec["knowledge"] = retrieved

    logging.info(f"[Pipeline] Phase 2: Generation (prompt-conditioned)")
    prompt_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    candidates = generator.generate_sequences(
        prompt_ids=prompt_ids,
        n_sequences=args.n_sequences,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        design_spec=design_spec,
    )
    logging.info(f"[Pipeline] Generated {len(candidates)} sequences")

    logging.info(f"[Pipeline] Phase 3: Evaluation (composite scoring)")
    cand_seqs = [c["sequence"] for c in candidates]
    full_results = evaluate_sequences(cand_seqs, known_amps=sequences[:10])

    logging.info(f"[Pipeline] Phase 4: Feedback & Closed-Loop Optimization")
    optimizer = FeedbackOptimizer()
    feedback = optimizer.generate_feedback(full_results, round_num=1)
    updated_spec = optimizer.apply_feedback_to_spec(design_spec, feedback)
    logging.info(f"[Pipeline] Feedback: {len(feedback['analysis']['suggestions'])} suggestions")
    for s in feedback['analysis']['suggestions']:
        logging.info(f"  - {s}")
    logging.info(f"[Pipeline] Updated spec: {updated_spec.get('preference', 'N/A')}")

    output_dir = args.output_dir or os.path.join("outputs", "pipeline")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pipeline_output = {
        "timestamp": timestamp,
        "config": vars(args),
        "training_history": history,
        "candidates": full_results,
        "feedback": feedback,
        "updated_design_spec": updated_spec,
    }
    output_path = os.path.join(output_dir, f"pipeline_result_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_output, f, ensure_ascii=False, indent=2)
    logging.info(f"[Pipeline] Full pipeline result saved to: {output_path}")

    return pipeline_output


def main():
    parser = argparse.ArgumentParser(description="KRPG Server-Side Full Deployment")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["train", "generate", "evaluate", "full", "pipeline"],
                        help="Execution mode")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to AMP sequence data (JSON)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input file for evaluation mode")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save checkpoints")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs")

    parser.add_argument("--d_model", type=int, default=256,
                        help="Transformer embedding dimension")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=512,
                        help="Feed-forward dimension")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum sequence length")
    parser.add_argument("--n_sequences", type=int, default=100,
                        help="Number of sequences to generate per target")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Generation temperature")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling")

    args = parser.parse_args()

    log_dir = os.path.join("logs", "server")
    logger = setup_logging(log_dir)
    device = get_device()

    logger.info(f"Arguments: {vars(args)}")

    if args.mode == "train":
        train_mode(args, device)
    elif args.mode == "generate":
        generate_mode(args, device)
    elif args.mode == "evaluate":
        evaluate_mode(args, device)
    elif args.mode in ("full", "pipeline"):
        full_pipeline_mode(args, device)
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
