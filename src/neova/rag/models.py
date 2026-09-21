from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    """One section of one public document. `text` is what the answering LLM reads; `search_text`
    is what the index matches against."""
    chunk_id: str
    doc_id: str
    title: str
    section_path: list[str]
    pages: list[int]
    audience: str
    statut: str                     # current | deprecated | unknown
    updated: str | None
    effective_from: str | None
    offer_window: dict | None       # {"from", "to"} of an offer's subscription period
    supersedes: str | None
    source_hash: str
    source_format: str              # pdf | image
    has_table: bool
    tables: list[dict]              # verbatim {"headers", "rows"}
    text: str
    search_text: str
    article_no: int | None = None
    references: list[str] = field(default_factory=list)
    referenced_by: list[str] = field(default_factory=list)
    unresolved_references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


@dataclass
class Corpus:
    chunks: list[Chunk]
    excluded: dict  # doc_id -> audience of every document kept out of the index
