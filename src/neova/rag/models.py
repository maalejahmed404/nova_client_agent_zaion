from dataclasses import dataclass, field


@dataclass
class Line:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    spans: list[tuple[str, bool]]

    @property
    def text(self) -> str:
        return ''.join(text for text, _ in self.spans).strip()

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class Table:
    page: int
    headers: list[str]
    rows: list[list[str]]
    line_ids: set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            'page': self.page,
            'headers': self.headers,
            'rows': self.rows,
            'line_ids': sorted(self.line_ids)
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Table':
        return cls(
            page=d['page'],
            headers=d['headers'],
            rows=d['rows'],
            line_ids=set(d['line_ids'])
        )


@dataclass
class Element:
    kind: str
    text: str
    level: int = 0
    marker: str | None = None
    table: Table | None = None
    pages: set[int] = field(default_factory=set)


@dataclass
class Document:
    doc_id: str
    source_format: str
    title: str
    audience: str
    statut: str
    updated: str | None = None
    effective_from: str | None = None
    offer_window: dict[str, str] | None = None
    supersedes: str | None = None
    preamble: str = ''
    elements: list[Element] = field(default_factory=list)
    source_hash: str = ''


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    heading: str
    text: str
    search_text: str
    tables: list[Table]
    pages: list[int]
    audience: str
    statut: str
    updated: str | None = None
    effective_from: str | None = None
    offer_window: dict[str, str] | None = None
    supersedes: str | None = None

    def to_dict(self) -> dict:
        return {
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'title': self.title,
            'heading': self.heading,
            'text': self.text,
            'search_text': self.search_text,
            'tables': [table.to_dict() for table in self.tables],
            'pages': self.pages,
            'audience': self.audience,
            'statut': self.statut,
            'updated': self.updated,
            'effective_from': self.effective_from,
            'offer_window': self.offer_window,
            'supersedes': self.supersedes
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Chunk':
        return cls(
            chunk_id=d['chunk_id'],
            doc_id=d['doc_id'],
            title=d['title'],
            heading=d['heading'],
            text=d['text'],
            search_text=d['search_text'],
            tables=[Table.from_dict(t) for t in d['tables']],
            pages=d['pages'],
            audience=d['audience'],
            statut=d['statut'],
            updated=d.get('updated'),
            effective_from=d.get('effective_from'),
            offer_window=d.get('offer_window'),
            supersedes=d.get('supersedes')
        )