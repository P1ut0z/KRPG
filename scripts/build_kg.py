"""
Build a KRPG knowledge graph from normalized AMP records.

Example:
  python scripts/build_kg.py --amp-csv data/amp_dataset_clean.csv --out data/knowledge_base
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from krpg.knowledge import KnowledgeGraph
from krpg.knowledge.ingestion import load_amp_csv_records


def main():
    parser = argparse.ArgumentParser(description="Build KRPG KG files from AMP data")
    parser.add_argument("--amp-csv", default=os.path.join("data", "amp_dataset_clean.csv"))
    parser.add_argument("--out", default=os.path.join("data", "knowledge_base"))
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--include-demo-rules", action="store_true", default=True)
    parser.add_argument("--dramp-tsv", default=os.path.join("data", "raw", "dramp", "general_amps.txt"))
    parser.add_argument("--dbaasp-dir", default=os.path.join("data", "raw", "dbaasp"))
    parser.add_argument("--no-raw-enrichment", action="store_true",
                        help="Skip DRAMP/DBAASP raw assay and metadata enrichment")
    args = parser.parse_args()

    kg = KnowledgeGraph(data_dir=args.out)
    if args.include_demo_rules:
        kg.build_default_amp_knowledge_graph()

    records = load_amp_csv_records(args.amp_csv, max_records=args.max_records)
    n_added = kg.build_from_amp_records(records, source=args.amp_csv)

    enrichment = {}
    if not args.no_raw_enrichment:
        enrichment["dramp"] = kg.enrich_from_dramp_tsv(args.dramp_tsv)
        enrichment["dbaasp"] = kg.enrich_from_dbaasp_index(args.dbaasp_dir)

    paths = kg.save_jsonl(args.out)

    summary = kg.summary()
    summary["peptides_added"] = n_added
    summary["enrichment"] = enrichment
    summary_path = os.path.join(args.out, "kg_summary.json")
    os.makedirs(args.out, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Built KG with {summary['total_entities']} entities and {summary['total_relations']} relations")
    print(f"Peptides added: {n_added}")
    print(f"Entities: {paths['entities']}")
    print(f"Relations: {paths['relations']}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
