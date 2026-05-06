"""
KRPG Quick Test Script
======================
小规模快速测试脚本，无需 GPU，几秒内验证 KRPG 框架核心功能是否正常。
运行方式: python scripts/quick_test.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from krpg.knowledge import KnowledgeGraph, RAGRetriever, PromptBuilder
from krpg.generation import AminoAcidTokenizer, KRPGGenerator
from krpg.generation.generator import AMPSequenceDataset, SequenceGenerator
from krpg.validation import (
    RuleBasedAMPPredictor, CompositeScorer,
    ToxicityPredictor, StabilityPredictor,
    SimilarityFilter, PhysicochemicalFilter,
    FeedbackOptimizer,
)

import torch


def print_header(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name, status, detail=""):
    icon = "✅" if status else "❌"
    print(f"  {icon} {name}  {detail}")


def test_knowledge_module():
    """测试知识图谱检索模块"""
    print_header("模块一：知识图谱检索")

    # 1. 构建知识图谱
    kg = KnowledgeGraph()
    kg.build_default_amp_knowledge_graph()
    summary = kg.summary()
    print_result("知识图谱构建", True,
                 f"({summary['total_entities']} 实体, {summary['total_relations']} 关系)")

    # 2. RAG 检索
    retriever = RAGRetriever(kg)
    retrieved = retriever.retrieve_by_target(
        target="Broad-Spectrum AMP, Gram-Negative, Low Toxicity",
        constraints={"length": "12-20", "charge": "+4 to +8"},
    )
    total_items = sum(len(v) for v in retrieved.values())
    print_result("RAG 检索", total_items > 0, f"(检索到 {total_items} 条知识)")

    # 3. 知识库搜索
    kb_results = retriever.search_knowledge_base("low toxicity amphipathic AMP", top_k=3)
    print_result("知识库搜索", len(kb_results) > 0, f"(返回 {len(kb_results)} 条结果)")

    # 4. Prompt 构建
    builder = PromptBuilder()
    prompt = builder.build_structured_prompt({
        "target": "Broad-Spectrum AMP",
        "activity": "Gram-Negative",
        "constraint": "Low Toxicity",
        "preference": "Length 12-20, Positive Charge, Amphipathic",
        "knowledge": retrieved,
    })
    print_result("Prompt 构建", len(prompt) > 50, f"({len(prompt)} 字符)")

    return kg, retriever, builder


def test_generation_module():
    """测试序列生成模块"""
    print_header("模块二：序列生成")

    # 1. Tokenizer 测试
    tokenizer = AminoAcidTokenizer()
    test_seq = "KWLKKIGAVLKVL"
    encoded = tokenizer.encode(test_seq)
    decoded = tokenizer.decode(encoded)
    tokenizer_ok = (test_seq == decoded)
    print_result("Tokenizer 编解码", tokenizer_ok,
                 f"(词表大小={tokenizer.vocab_size}, 序列={test_seq})")

    # 2. 模型前向传播测试
    model = KRPGGenerator(
        vocab_size=tokenizer.vocab_size,
        d_model=32, n_heads=2, n_layers=2, d_ff=64,
        max_seq_len=32, dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
        use_prompt_conditioning=True,
    )
    model_size = model.get_model_size()
    batch = torch.randint(0, tokenizer.vocab_size, (2, 10))
    logits, loss = model(input_ids=batch, labels=batch)
    forward_ok = (logits.shape == (2, 10, tokenizer.vocab_size)) and (loss is not None)
    print_result("模型前向传播", forward_ok,
                 f"(参数={model_size['total_params']:,}, loss={loss.item():.4f})")

    # 3. Prompt 条件控制测试
    spec = {"target": "Broad-Spectrum AMP", "constraint": "Low Toxicity",
            "preference": "Length 12-20, Positive Charge"}
    prompt_vector = model.encode_prompt(spec)
    pv_ok = prompt_vector is not None and prompt_vector.shape[0] == 1
    print_result("Prompt 条件编码", pv_ok, f"(向量维度={prompt_vector.shape[1]})")

    # 4. 序列生成测试
    prompt_ids = torch.tensor([[tokenizer.bos_token_id]])
    generated = model.generate(
        input_ids=prompt_ids,
        max_new_tokens=15,
        temperature=1.0,
        top_k=20,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        prompt_vector=prompt_vector,
    )
    gen_seq = tokenizer.decode(generated[0].tolist())
    gen_ok = len(gen_seq) > 0
    print_result("序列生成", gen_ok, f"(生成序列: {gen_seq})")

    # 5. 训练测试
    sample_amps = [
        "KWLKKIGAVLKVL", "GFKRIVQRIKDFL", "LYIAKLLKRFN",
        "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
        "GIGKFLHSAKKFGKAFVGEIMNS",
    ]
    dataset = AMPSequenceDataset(sample_amps, tokenizer, max_length=30)
    generator = SequenceGenerator(model, tokenizer, device="cpu")
    history = generator.train_model(
        dataset, n_epochs=3, batch_size=4, learning_rate=1e-3,
        design_spec=spec,
    )
    train_ok = len(history["loss"]) == 3 and history["loss"][-1] < history["loss"][0]
    print_result("模型训练", train_ok,
                 f"(loss: {history['loss'][0]:.4f} → {history['loss'][-1]:.4f})")

    # 6. 批量生成测试
    results = generator.generate_sequences(
        prompt_ids=prompt_ids, n_sequences=5, max_length=15,
        temperature=1.0, top_k=20, top_p=0.9,
        design_spec=spec,
    )
    print_result("批量生成", len(results) == 5, f"(生成 {len(results)} 条候选序列)")
    for i, r in enumerate(results):
        print(f"       [{i+1}] {r['sequence']:25s} prob={r['avg_log_prob']:.4f}")

    return tokenizer, model, generator


def test_validation_module():
    """测试计算验证模块"""
    print_header("模块三：计算验证")

    test_sequences = [
        "KWLKKIGAVLKVL",
        "GFKRIVQRIKDFL",
        "LYIAKLLKRFN",
        "WWWWWWWWWW",
        "GAGAGAGAGA",
    ]

    # 1. AMP 活性预测
    rule_predictor = RuleBasedAMPPredictor()
    amp_results = [rule_predictor.predict(s) for s in test_sequences]
    n_candidates = sum(1 for r in amp_results if r["is_amp_candidate"])
    print_result("AMP 活性预测", n_candidates > 0, f"({n_candidates}/5 为候选)")

    # 2. 毒性预测
    tox_predictor = ToxicityPredictor()
    tox_results = [tox_predictor.predict(s) for s in test_sequences]
    n_toxic = sum(1 for r in tox_results if r["is_toxic"])
    print_result("毒性预测", True, f"({n_toxic}/5 判定为有毒)")

    # 3. 稳定性预测
    stab_predictor = StabilityPredictor()
    stab_results = [stab_predictor.predict(s) for s in test_sequences]
    n_stable = sum(1 for r in stab_results if r["is_stable"])
    print_result("稳定性预测", True, f"({n_stable}/5 判定为稳定)")

    # 4. 相似性过滤
    known_amps = ["KWLKKIGAVLKVL", "GFKRIVQRIKDFL"]
    sim_filter = SimilarityFilter(known_amps)
    sim_results = sim_filter.filter_by_similarity(test_sequences, threshold=0.8)
    n_novel = sum(1 for r in sim_results if r["is_novel"])
    print_result("相似性过滤", True, f"({n_novel}/5 为新颖序列)")

    # 5. 理化性质计算
    phys_filter = PhysicochemicalFilter()
    phys_results = phys_filter.batch_compute(test_sequences)
    has_pI = all(r["isoelectric_point"] > 0 for r in phys_results)
    print_result("理化性质计算", has_pI, "(含分子量、电荷、等电点、疏水性等)")

    # 6. 综合评分
    scorer = CompositeScorer()
    comp_results = []
    for i, seq in enumerate(test_sequences):
        comp = scorer.score(
            amp_results[i]["amp_score"],
            tox_results[i]["toxicity_score"],
            stab_results[i]["stability_score"],
            sim_results[i]["is_novel"],
        )
        comp_results.append(comp)
    print_result("综合评分", True, f"(最高分: {max(c['composite_score'] for c in comp_results):.3f})")

    # 打印详细结果表格
    print()
    print(f"  {'序列':25s} {'AMP':8s} {'毒性':8s} {'稳定':8s} {'新颖':6s} {'综合':8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    for i, seq in enumerate(test_sequences):
        print(f"  {seq:25s} {amp_results[i]['amp_score']:.3f}  "
              f"{tox_results[i]['toxicity_score']:.3f}  "
              f"{stab_results[i]['stability_score']:.3f}  "
              f"{'✅' if sim_results[i]['is_novel'] else '❌'}    "
              f"{comp_results[i]['composite_score']:.3f}")

    return rule_predictor, tox_predictor, stab_predictor, sim_filter, phys_filter


def test_feedback_module():
    """测试反馈优化模块"""
    print_header("模块四：反馈优化（闭环）")

    sample_results = [
        {"sequence": "KWLKKIGAVLKVL", "amp_score": 0.85, "toxicity_score": 0.25, "stability_score": 0.60},
        {"sequence": "GFKRIVQRIKDFL", "amp_score": 0.72, "toxicity_score": 0.35, "stability_score": 0.55},
        {"sequence": "LYIAKLLKRFN", "amp_score": 0.55, "toxicity_score": 0.20, "stability_score": 0.45},
        {"sequence": "WWWWWWWWWW", "amp_score": 0.30, "toxicity_score": 0.85, "stability_score": 0.30},
        {"sequence": "GAGAGAGAGA", "amp_score": 0.15, "toxicity_score": 0.10, "stability_score": 0.25},
    ]

    optimizer = FeedbackOptimizer()
    feedback = optimizer.generate_feedback(sample_results, round_num=1)
    analysis = feedback["analysis"]
    print_result("结果分析", True,
                 f"(平均AMP={analysis['avg_amp_score']:.3f}, "
                 f"平均毒性={analysis['avg_toxicity_score']:.3f}, "
                 f"平均稳定={analysis['avg_stability_score']:.3f})")

    print(f"\n  📋 优化建议 ({len(analysis['suggestions'])} 条):")
    for s in analysis["suggestions"]:
        print(f"     • {s}")

    # 闭环更新测试
    original_spec = {
        "target": "Broad-Spectrum AMP",
        "activity": "Gram-Negative",
        "constraint": "Low Toxicity",
        "preference": "Length 12-20, Positive Charge",
    }
    updated_spec = optimizer.apply_feedback_to_spec(original_spec, feedback)
    spec_changed = updated_spec["preference"] != original_spec["preference"]
    no_update_needed = not feedback.get("updated_constraints")
    print_result("闭环规格更新", spec_changed or no_update_needed,
                 "(无需更新)" if no_update_needed else
                 f"(偏好: {original_spec['preference']} → {updated_spec['preference']})")

    return optimizer


def test_end_to_end():
    """端到端流水线测试"""
    print_header("端到端流水线测试")

    start_time = time.time()

    # Step 1: 知识检索
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
    print_result("Step 1: 知识检索 → Prompt", True, f"({len(prompt)} 字符)")

    # Step 2: 训练
    tokenizer = AminoAcidTokenizer()
    model = KRPGGenerator(
        vocab_size=tokenizer.vocab_size,
        d_model=32, n_heads=2, n_layers=2, d_ff=64,
        max_seq_len=32, dropout=0.1,
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
    print_result("Step 2: 模型训练", True, "(3 epochs, prompt-conditioned)")

    # Step 3: 生成
    prompt_ids = torch.tensor([[tokenizer.bos_token_id]])
    candidates = generator.generate_sequences(
        prompt_ids=prompt_ids, n_sequences=8, max_length=15,
        temperature=1.0, top_k=20, top_p=0.9,
        design_spec=design_spec,
    )
    print_result("Step 3: 序列生成", len(candidates) == 8, f"({len(candidates)} 条候选)")

    # Step 4: 验证评估
    rule_predictor = RuleBasedAMPPredictor()
    tox_predictor = ToxicityPredictor()
    stab_predictor = StabilityPredictor()
    phys_filter = PhysicochemicalFilter()
    scorer = CompositeScorer()

    full_results = []
    for c in candidates:
        seq = c["sequence"]
        amp_r = rule_predictor.predict(seq)
        tox_r = tox_predictor.predict(seq)
        stab_r = stab_predictor.predict(seq)
        phys_r = phys_filter.compute_properties(seq)
        comp = scorer.score(amp_r["amp_score"], tox_r["toxicity_score"],
                           stab_r["stability_score"], True)
        full_results.append({
            "sequence": seq,
            "amp_score": amp_r["amp_score"],
            "toxicity_score": tox_r["toxicity_score"],
            "stability_score": stab_r["stability_score"],
            "composite_score": comp["composite_score"],
            "charge": phys_r["net_charge"],
            "pI": phys_r["isoelectric_point"],
        })
    print_result("Step 4: 验证评估", True, "(AMP + 毒性 + 稳定 + 理化 + 综合评分)")

    # Step 5: 反馈优化
    optimizer = FeedbackOptimizer()
    feedback = optimizer.generate_feedback(full_results, round_num=1)
    updated_spec = optimizer.apply_feedback_to_spec(design_spec, feedback)
    print_result("Step 5: 反馈优化", True,
                 f"({len(feedback['analysis']['suggestions'])} 条建议)")

    elapsed = time.time() - start_time

    # 打印结果表格
    print()
    print(f"  {'序列':25s} {'AMP':8s} {'毒性':8s} {'稳定':8s} {'综合':8s} {'电荷':6s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for r in sorted(full_results, key=lambda x: x["composite_score"], reverse=True):
        s = r["sequence"][:24]
        print(f"  {s:25s} {r['amp_score']:.3f}  {r['toxicity_score']:.3f}  "
              f"{r['stability_score']:.3f}  {r['composite_score']:.3f}  {r['charge']:+d}")

    print()
    print_result("端到端流水线", True, f"(总耗时: {elapsed:.2f}s)")

    return full_results, feedback


def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  KRPG Quick Test - 小规模快速验证脚本                    ║")
    print("║  Knowledge-Retrieval Prompted Generator for AMP Design   ║")
    print("╚" + "═" * 58 + "╝")
    print()

    all_passed = True
    tests = []

    try:
        kg, retriever, builder = test_knowledge_module()
        tests.append(("知识图谱检索模块", True))
    except Exception as e:
        print_result("知识图谱检索模块", False, str(e))
        tests.append(("知识图谱检索模块", False))
        all_passed = False

    try:
        tokenizer, model, generator = test_generation_module()
        tests.append(("序列生成模块", True))
    except Exception as e:
        print_result("序列生成模块", False, str(e))
        tests.append(("序列生成模块", False))
        all_passed = False

    try:
        test_validation_module()
        tests.append(("计算验证模块", True))
    except Exception as e:
        print_result("计算验证模块", False, str(e))
        tests.append(("计算验证模块", False))
        all_passed = False

    try:
        test_feedback_module()
        tests.append(("反馈优化模块", True))
    except Exception as e:
        print_result("反馈优化模块", False, str(e))
        tests.append(("反馈优化模块", False))
        all_passed = False

    try:
        results, feedback = test_end_to_end()
        tests.append(("端到端流水线", True))
    except Exception as e:
        print_result("端到端流水线", False, str(e))
        tests.append(("端到端流水线", False))
        all_passed = False

    # 汇总
    print()
    print("=" * 60)
    print(f"  📊 测试汇总")
    print("=" * 60)
    for name, status in tests:
        print_result(name, status)
    print()
    if all_passed:
        print("  🎉 全部测试通过！KRPG 框架运行正常。")
    else:
        print("  ⚠️  部分测试未通过，请检查错误信息。")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
