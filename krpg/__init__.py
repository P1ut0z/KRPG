__version__ = "0.1.0"

__all__ = [
    "KnowledgeGraph",
    "RAGRetriever",
    "PromptBuilder",
    "AminoAcidTokenizer",
    "KRPGGenerator",
    "AMPPredictor",
    "ToxicityPredictor",
    "StabilityPredictor",
    "SimilarityFilter",
    "FeedbackOptimizer",
]


def __getattr__(name):
    if name in {"KnowledgeGraph", "RAGRetriever", "PromptBuilder"}:
        from krpg import knowledge
        return getattr(knowledge, name)
    if name in {"AminoAcidTokenizer", "KRPGGenerator"}:
        from krpg import generation
        return getattr(generation, name)
    if name in {"AMPPredictor", "ToxicityPredictor", "StabilityPredictor", "SimilarityFilter", "FeedbackOptimizer"}:
        from krpg import validation
        return getattr(validation, name)
    raise AttributeError(f"module 'krpg' has no attribute {name!r}")
