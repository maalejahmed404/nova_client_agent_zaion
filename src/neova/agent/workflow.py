"""Workflow decisions of the graph that are not in any document: which sections an intent always
needs. Read by the graph, never by ingestion."""
from functools import lru_cache
from pathlib import Path

import yaml

from neova.rag.ingest import slug

SPEC_PATH = Path(__file__).with_name("workflow_spec.yaml")


@lru_cache
def spec() -> dict:
    return yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


def intents() -> list[str]:
    return list(spec()["required_evidence"])


def required_chunk_ids(intent: str) -> list[str]:
    return [f"{t['doc']}#{slug(t['heading'])}" for t in spec()["required_evidence"].get(intent, [])]
