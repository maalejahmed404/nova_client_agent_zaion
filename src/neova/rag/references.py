import json
import re

from neova.config import CACHE_DIR
from neova.rag.models import Chunk


def formal_references(chunks: list[Chunk]) -> dict[str, list[dict]]:
    article_chunks: dict[str, Chunk] = {}
    for chunk in chunks:
        if chunk.heading.startswith("Article "):
            article_chunks[chunk.heading.split()[1]] = chunk

    pattern = re.compile(r"\b(?:voir\s+CGV,\s+)?articles?\s+(\d+)", re.IGNORECASE)
    result: dict[str, list[dict]] = {}
    for chunk in chunks:
        refs: list[dict] = []
        seen: set[str] = set()

        text_blocks = chunk.text.split("# texte\n")
        if len(text_blocks) < 2:
            continue
        content = text_blocks[1]

        for match in pattern.finditer(content):
            anchor = match.group()
            number = match.group(1)
            if number in article_chunks:
                target = article_chunks[number]
                if target.chunk_id == chunk.chunk_id:
                    continue
                if anchor not in seen:
                    refs.append(
                        {"anchor": anchor.strip(), "target": target.chunk_id, "source": "regex"}
                    )
                    seen.add(anchor)

        if refs:
            result[chunk.chunk_id] = refs

    return result


def build_references(chunks: list[Chunk]) -> None:
    ref_map = formal_references(chunks)
    path = CACHE_DIR / "corpus" / "ref_map.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ref_map, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    corpus_path = CACHE_DIR / "corpus" / "corpus.json"
    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)

    chunks = [Chunk.from_dict(d) for d in data]
    build_references(chunks)

    ref_map = formal_references(chunks)
    for citing_id, refs in ref_map.items():
        for ref in refs:
            print(f"{citing_id} -> {ref['target']}  ({ref['anchor']})")
