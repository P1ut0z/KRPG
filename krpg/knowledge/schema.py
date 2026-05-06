from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Evidence:
    """Provenance for a KG fact or design rule."""

    source: str
    reference_id: Optional[str] = None
    url: Optional[str] = None
    confidence: float = 1.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntityRecord:
    """Typed KG entity used by retrieval and prompt construction."""

    id: str
    type: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationRecord:
    """Typed KG relation with optional evidence and confidence."""

    head: str
    relation: str
    tail: str
    properties: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0

    def to_tuple(self):
        props = dict(self.properties)
        props.setdefault("evidence", self.evidence)
        props.setdefault("confidence", self.confidence)
        return self.head, self.relation, self.tail, props


ENTITY_TYPES = {
    "peptide",
    "target_organism",
    "target_class",
    "pathogen_class",
    "activity_assay",
    "toxicity_assay",
    "activity_class",
    "physicochemical_property",
    "motif",
    "structure_feature",
    "mechanism",
    "go_term",
    "literature_reference",
    "database_record",
    "design_rule",
    "property",
    "pathogen",
    "source_organism",
    "modification",
}


RELATION_TYPES = {
    "active_against",
    "has_activity",
    "has_activity_class",
    "has_activity_assay",
    "has_toxicity_assay",
    "has_property",
    "has_motif",
    "has_structure",
    "has_mechanism",
    "mapped_to",
    "supported_by",
    "derived_from",
    "has_source_organism",
    "has_modification",
    "constrains",
    "suggests",
    "warns_against",
    "associated_with",
    "related_to",
    "promotes",
}
