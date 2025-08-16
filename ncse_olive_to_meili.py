#!/usr/bin/env python3
"""
NCSE (Olive/XMD) → Meilisearch JSONL (chunked)

Fixes:
- Strip page header/footer stubs like "( 119 )" from text.
- Track both image sequence page numbers (PAGE_NO) and printed page labels (PAGE_LABEL):
  page_no_start/end, page_label_start/end.
- Normalize section: "None" -> null.
- Best TOC entry across chain, title cleaning + first-sentence fallback,
  continuation stitching, chunking at paragraph boundaries, Meilisearch-safe IDs.

Usage:
  python3 ncse_olive_to_meili.py "/path/to/English_Woman's_Journal" -o ejw_olive_chunks.jsonl --chunk-size 2200
"""

import argparse
import calendar
import datetime
import io
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

# ---------- normalization & helpers ----------

WS_RE = re.compile(r"\s+")
SMART_SUBS = {
    "\u2018": "'", "\u2019": "'",
    "\u201C": '"', "\u201D": '"',
    "\u2013": "-", "\u2014": "-",
    "\ufeff": ""
}
# Olive oddballs & box-drawing: map to space or dash
ODD_GLYPHS = {
    "¦": "-", "�": "", "•": "-", "–": "-", "—": "-",
    "·": " ", "∙": " ", "●": " ",
    "”": '"', "“": '"', "’": "'", "‘": "'"
}

MULTI_SPACE_RE = re.compile(r"\s+")
PUNKT_FIX = re.compile(r"\s+([,.;:!?])")
TITLE_LEAD_NUM_RE = re.compile(r"^\s*[\(\[]?\s*\d{1,4}\s*[\)\]]?\s+")
TITLE_TRAIL_NUM_RE = re.compile(r"\s+[—\-–]?\s*\d{1,4}\s*$")

HEADER_STUB_RE = re.compile(r"^\s*\(?\s*\d{1,4}\s*\)?\s*$")  # e.g. "( 119 )" or "119"
ORNAMENT_RE = re.compile(r"^[\W_]{1,}$")  # lines of punctuation/rules only

def norm(s: str) -> str:
    if not s:
        return ""
    for k, v in SMART_SUBS.items():
        s = s.replace(k, v)
    for k, v in ODD_GLYPHS.items():
        s = s.replace(k, v)
    s = WS_RE.sub(" ", s).strip()
    return s

def tidy_punct(s: str) -> str:
    return PUNKT_FIX.sub(r"\1", s)

SMALL_WORDS = {"a","an","and","as","at","but","by","for","in","into","of","on","or","the","to","with"}
def smart_titlecase(s: str) -> str:
    parts = s.strip().split()
    if not parts:
        return s
    out = []
    for i, w in enumerate(parts):
        lw = w.lower()
        if i not in (0, len(parts) - 1) and lw in SMALL_WORDS:
            out.append(lw)
        else:
            out.append(w[:1].upper() + w[1:].lower())
    return " ".join(out)

def clean_toc_title(raw: str) -> str:
    if not raw:
        return raw
    t = norm(raw)
    t = MULTI_SPACE_RE.sub(" ", t).strip()
    t = tidy_punct(t)
    t = TITLE_LEAD_NUM_RE.sub("", t)
    t = TITLE_TRAIL_NUM_RE.sub("", t)
    t = t.strip("()[]{} .,:;–—-")
    if len(t) > 5 and t.upper() == t:
        t = smart_titlecase(t)
    return t.strip()

def looks_corrupt(title: str) -> bool:
    if not title:
        return True
    t = title.strip()
    if len(t) < 6:
        return True
    letters = sum(ch.isalpha() for ch in t)
    digits = sum(ch.isdigit() for ch in t)
    symbols = sum((not ch.isalnum()) and (not ch.isspace()) for ch in t)
    if letters < 6:
        return True
    if digits >= 2 or symbols > max(2, letters // 4):
        return True
    return False

def title_from_text(paragraphs, min_len=24, max_len=120):
    if not paragraphs:
        return None
    # Try first few paragraphs in case the first is a header or debris
    for first in paragraphs[:3]:
        cand_src = first.strip()
        if sum(c.isalpha() for c in cand_src) < 6:
            continue
        end = re.search(r"[.!?]\s", cand_src)
        cand = cand_src[: end.end()].strip() if end else cand_src[: max_len].strip()
        if len(cand) < min_len and len(cand_src) > min_len:
            cand = cand_src[: max_len].strip()
        cand = cand.rstrip(",;: ").strip()
        if not looks_corrupt(cand):
            return cand
    return None

def month_str_to_num(m):
    m = m.strip().lower()
    if m.isdigit():
        mi = int(m)
        return mi if 1 <= mi <= 12 else None
    for i in range(1, 13):
        if m == calendar.month_name[i].lower() or m == calendar.month_abbr[i].lower():
            return i
    return None

def ydir_is_year(name: str) -> bool:
    return name.isdigit() and len(name) == 4

# ---------- TOC parsing ----------

def parse_toc(toc_path: Path):
    """
    Parse Olive TOC:
      Head_np/Meta/Application_Data/Application_Info[@AI_TYPE='kc:volume'|'kc:number']
      Body_np/Section/Page/Entity[ @ID, @NAME (title), @PAGE_NO, CONTINUATION_*]
    Returns dict:
      {
        "_issue": {"volume": "...", "number": "..."},
        "Ar00100": {"title": "...", "first_page":"1", "section":"Front page", "CONTINUATION_TO":"...", "CONTINUATION_FROM":"..."},
        ...
      }
    """
    out = {}
    if not toc_path.exists():
        return out
    try:
        tree = ET.parse(str(toc_path))
        root = tree.getroot()

        issue_meta = {"volume": None, "number": None}
        for app in root.findall(".//Head_np/Meta/Application_Data/Application_Info"):
            aitype = (app.attrib.get("AI_TYPE") or "").strip()
            if aitype in ("kc:volume", "kc:number"):
                item = app.find("Ai_Item")
                if item is not None and item.attrib.get("NAME"):
                    if aitype == "kc:volume":
                        issue_meta["volume"] = item.attrib["NAME"].strip()
                    elif aitype == "kc:number":
                        issue_meta["number"] = item.attrib["NAME"].strip()
        out["_issue"] = issue_meta

        for sec in root.findall(".//Body_np/Section"):
            section_name = sec.attrib.get("NAME", "").strip()
            for page in sec.findall("./Page"):
                page_no = (page.attrib.get("PAGE_NO") or "").strip()
                for ent in page.findall("./Entity"):
                    if ent.attrib.get("ENTITY_TYPE") != "Article":
                        continue
                    arid = ent.attrib.get("ID")
                    if not arid:
                        continue
                    raw_title = ent.attrib.get("NAME", "")
                    out[arid] = {
                        "title": norm(raw_title),
                        "first_page": page_no,
                        "section": section_name,
                        "CONTINUATION_TO": ent.attrib.get("CONTINUATION_TO"),
                        "CONTINUATION_FROM": ent.attrib.get("CONTINUATION_FROM"),
                    }
    except Exception as e:
        print(f"[WARN] TOC parse failed {toc_path}: {e}")
    return out

# ---------- Article parsing (Ar*.xml inside Document.zip) ----------

def extract_article_text(xmd_bytes: bytes):
    """
    Parse an Olive XMD article (ArXXXXX.xml):
      - Root: <XMD-entity ...> + <Meta ...> + <Link ...> + <Content><Primitive>…</Primitive></Content>
      - Prefer normalized words QW; else W; keep punctuation from W; ignore layout L.
      - Each Primitive becomes one paragraph (then we filter page-label stubs).
    Returns: meta(dict), paragraphs(list[str]), links(dict)
    """
    root = ET.fromstring(xmd_bytes)  # encoding handled by XML declaration
    if root.tag != "XMD-entity":
        raise ValueError(f"unexpected root tag {root.tag}")
    meta = dict(root.attrib)
    meta_el = root.find("Meta")
    if meta_el is not None:
        for k, v in meta_el.attrib.items():
            if k not in meta or not meta[k]:
                meta[k] = v

    link = root.find("Link")
    links = {
        "NEXT_ID": link.get("NEXT_ID") if link is not None else None,
        "PREV_ID": link.get("PREV_ID") if link is not None else None,
        "CONTINUATION_TO": meta.get("CONTINUATION_TO"),
        "CONTINUATION_FROM": meta.get("CONTINUATION_FROM"),
    }

    # collect paragraphs
    paragraphs = []
    content = root.find("Content")
    if content is not None:
        for prim in content.findall("Primitive"):
            tokens = []
            for child in prim:
                tag = child.tag
                text = (child.text or "").strip()
                if not text:
                    continue
                if tag == "QW":
                    tokens.append(text)
                elif tag in ("Q", "q"):
                    continue  # covered by QW
                elif tag == "W":
                    tokens.append(text)
                elif tag == "S":
                    w = child.find("W")
                    if w is not None and (w.text or "").strip():
                        tokens.append(w.text.strip())
                # ignore L and others
            para = norm(" ".join(tokens))
            if para:
                paragraphs.append(para)

    # filter header/footer stubs (page labels / ornaments) at paragraph boundaries
    page_label = (meta.get("PAGE_LABEL") or "").strip()
    wordcnt = int(meta.get("WORDCNT") or "0")
    cleaned = []
    for i, para in enumerate(paragraphs):
        if i == 0:
            if (wordcnt <= 3 and (HEADER_STUB_RE.match(para) or ORNAMENT_RE.match(para))) \
               or (page_label and para.replace(" ", "") in (page_label, f"({page_label})", f"[{page_label}]")):
                continue
        cleaned.append(para)
    paragraphs = cleaned

    return meta, paragraphs, links

def extract_page_meta(xmd_bytes: bytes):
    """Parse an Olive XMD page (PgXXX.xml) and return metadata.

    Expected structure:
      <XMD-PAGE PAGE_NO="..." PAGE_LABEL="...">
        ...
        <Entity ID="Ar00123" ENTITY_TYPE="Article" />
      </XMD-PAGE>

    Returns dict with PAGE_NO, PAGE_LABEL and list of article IDs appearing on
    the page. Raises ValueError if the root element isn't an XMD-PAGE.
    """
    root = ET.fromstring(xmd_bytes)
    if root.tag != "XMD-PAGE":
        raise ValueError(f"unexpected root tag {root.tag}")
    page_no = root.attrib.get("PAGE_NO")
    page_label = root.attrib.get("PAGE_LABEL")
    articles = []
    for ent in root.findall(".//Entity"):
        if ent.attrib.get("ENTITY_TYPE") == "Article" and ent.attrib.get("ID"):
            articles.append(ent.attrib["ID"])
    return {"PAGE_NO": page_no, "PAGE_LABEL": page_label, "articles": articles}

def month_dir_to_int(name: str) -> int | None:
    try:
        return int(name)
    except ValueError:
        return month_str_to_num(name)

def ymd_from_issue_path(issue_dir: Path):
    try:
        y = int(issue_dir.parent.parent.name)
        m = month_dir_to_int(issue_dir.parent.name)
        d = int(issue_dir.name)
        if not m:
            return None, None
        return f"{y:04d}-{m:02d}-{d:02d}", y
    except Exception:
        return None, None

def collect_articles(issue_dir: Path):
    """Extract article texts and page metadata from an issue directory.

    Returns (articles, page_map) where page_map maps article IDs to lists of
    page info dicts (PAGE_NO/PAGE_LABEL) sorted by PAGE_NO.
    """
    zpath = issue_dir / "Document.zip"
    if not zpath.exists():
        return {}, {}
    articles = {}
    page_map: dict[str, list[dict[str, str | None]]] = {}
    with zipfile.ZipFile(zpath, "r") as zf:
        for n in zf.namelist():
            try:
                if re.search(r"/Ar\d{5}\.xml$", n):
                    meta, paras, links = extract_article_text(zf.read(n))
                    arid = meta.get("ID") or Path(n).stem
                    articles[arid] = {
                        "meta": meta,
                        "paras": paras,
                        "links": links,
                        "zip_member": n,
                    }
                elif re.search(r"/Pg\d{3}\.xml$", n):
                    pmeta = extract_page_meta(zf.read(n))
                    for arid in pmeta["articles"]:
                        page_map.setdefault(arid, []).append({
                            "PAGE_NO": pmeta["PAGE_NO"],
                            "PAGE_LABEL": pmeta["PAGE_LABEL"],
                        })
                else:
                    continue  # unknown type
            except Exception as e:
                print(f"[WARN] parse {zpath}::{n}: {e}")
    for arid, plist in page_map.items():
        plist.sort(key=lambda x: int(x["PAGE_NO"]) if (x["PAGE_NO"] or "").isdigit() else 0)
    return articles, page_map

def build_chains(articles: dict):
    """
    Build continuation chains using CONTINUATION_TO (preferred) or NEXT_ID.
    Returns list of chains (each is list[ArID]) preserving order.
    """
    next_map = {}
    prev_targets = set()
    for arid, a in articles.items():
        nxt = a["links"].get("CONTINUATION_TO") or a["links"].get("NEXT_ID")
        if nxt:
            next_map[arid] = nxt
            prev_targets.add(nxt)

    starts = [arid for arid in articles.keys() if arid not in prev_targets]
    chains, seen = [], set()

    for start in starts:
        cur = start
        chain = []
        while cur and cur in articles and cur not in seen:
            chain.append(cur)
            seen.add(cur)
            cur = next_map.get(cur)
        if chain:
            chains.append(chain)

    for arid in articles.keys():
        if arid not in seen:
            chains.append([arid])
    return chains

def pick_best_toc_for_chain(chain, toc_map):
    """
    Return (raw_title, section, first_page) choosing the first TOC entry
    whose cleaned title isn't corrupt; else fall back to any TOC title; else Nones.
    """
    for arid in chain:
        info = toc_map.get(arid)
        if not info:
            continue
        raw = info.get("title") or ""
        cleaned = clean_toc_title(raw)
        if cleaned and not looks_corrupt(cleaned):
            return raw, info.get("section"), info.get("first_page")
    for arid in chain:
        info = toc_map.get(arid)
        if not info:
            continue
        raw = info.get("title") or ""
        if raw:
            return raw, info.get("section"), info.get("first_page")
    return None, None, None

# ---------- Main driver ----------

def main():
    ap = argparse.ArgumentParser(description="NCSE Olive/XMD → Meilisearch JSONL (chunked).")
    ap.add_argument("edition_root", help="Folder containing YEAR/MONTH/DAY (monthly issues folder)")
    ap.add_argument("-o", "--out", default="ejw_olive_chunks.jsonl")
    ap.add_argument("--chunk-size", type=int, default=2200)
    ap.add_argument("--journal", default="English Woman’s Journal")
    ap.add_argument("--journal_id", default="english-womans-journal")
    args = ap.parse_args()

    root = Path(args.edition_root)
    outp = Path(args.out)

    total_issues = total_articles = total_chunks = 0

    with outp.open("w", encoding="utf-8") as outfh:
        for ydir in sorted([p for p in root.iterdir() if p.is_dir() and ydir_is_year(p.name)]):
            for mdir in sorted([p for p in ydir.iterdir() if p.is_dir()]):
                for ddir in sorted([p for p in mdir.iterdir() if p.is_dir()]):
                    z = ddir / "Document.zip"
                    if not z.exists():
                        continue
                    total_issues += 1

                    date_iso, year_val = ymd_from_issue_path(ddir)
                    toc_map = parse_toc(ddir / "TOC.xml")
                    issue_info = toc_map.get("_issue", {})
                    issue_volume = issue_info.get("volume")
                    issue_number = issue_info.get("number")

                    articles, page_map = collect_articles(ddir)
                    if not articles:
                        continue
                    chains = build_chains(articles)

                    for chain in chains:
                        first = articles[chain[0]]
                        m0 = first["meta"]
                        arid_parent = chain[0]

                        # title/section/page_start from best TOC entry in chain
                        raw_title, section, page_start_seq = pick_best_toc_for_chain(chain, toc_map)
                        if not raw_title:
                            raw_title = m0.get("NAME") or m0.get("LOGICAL_NAME") or ""
                        title_clean = clean_toc_title(raw_title) if raw_title else ""

                        # normalize section
                        if section:
                            section = section.strip()
                            if section.lower() == "none":
                                section = None

                        # author (rare in EJW Olive)
                        author = None

                        # date normalization
                        date_raw = m0.get("ISSUE_DATE") or ""
                        date_final = date_iso
                        try:
                            if date_raw:
                                d = datetime.datetime.strptime(date_raw, "%d/%m/%Y")
                                date_final = d.strftime("%Y-%m-%d")
                        except Exception:
                            pass
                        year_final = int(date_final[:4]) if date_final else year_val

                        # collect paragraphs & track start/end page numbers and labels
                        paragraphs = []
                        page_no_start = page_label_start = None
                        page_no_end = page_label_end = None

                        for i, arid in enumerate(chain):
                            a = articles[arid]
                            paragraphs.extend(a["paras"])
                            meta_i = a["meta"]
                            pinfo = page_map.get(arid)

                            if i == 0:
                                page_no_start = meta_i.get("PAGE_NO") or (
                                    pinfo[0]["PAGE_NO"] if pinfo else None
                                )
                                page_label_start = meta_i.get("PAGE_LABEL") or (
                                    pinfo[0]["PAGE_LABEL"] if pinfo else None
                                )

                            last_no = meta_i.get("PAGE_NO")
                            last_label = meta_i.get("PAGE_LABEL")
                            if pinfo:
                                if pinfo[-1].get("PAGE_NO"):
                                    last_no = pinfo[-1]["PAGE_NO"]
                                if pinfo[-1].get("PAGE_LABEL"):
                                    last_label = pinfo[-1]["PAGE_LABEL"]
                            if last_no:
                                page_no_end = last_no
                            if last_label:
                                page_label_end = last_label

                        if not page_no_start and page_start_seq:
                            page_no_start = page_start_seq

                        # finalize title (fallback to first sentence if TOC looks junky)
                        final_title = title_clean
                        if looks_corrupt(final_title):
                            alt = title_from_text(paragraphs)
                            if alt and not looks_corrupt(alt):
                                final_title = alt
                        display_title = final_title if final_title else None
                        raw_title_out = raw_title if raw_title else None

                        # chunk & write
                        chunk_idx, cur, total_len = 0, [], 0
                        parent_id = f"{args.journal_id}_{date_final}_{arid_parent}"  # Meilisearch-safe
                        for ptxt in paragraphs:
                            if not ptxt:
                                continue
                            if total_len + len(ptxt) > args.chunk_size and cur:
                                text = tidy_punct("\n\n".join(cur))
                                doc = {
                                    "id": f"{parent_id}_c{chunk_idx:03d}",
                                    "parent_id": parent_id,
                                    "chunk_index": chunk_idx,
                                    "article_id": arid_parent,
                                    "journal": args.journal,
                                    "journal_id": args.journal_id,
                                    "title": display_title,
                                    "raw_title": raw_title_out,
                                    "author": author,
                                    "section": section,
                                    "volume": issue_volume,
                                    "number": issue_number,
                                    "date": date_final,
                                    "year": year_final,
                                    "page_no_start": page_no_start,
                                    "page_no_end": page_no_end,
                                    "page_label_start": page_label_start,
                                    "page_label_end": page_label_end,
                                    "text": text,
                                }
                                outfh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                                chunk_idx += 1
                                cur, total_len = [ptxt], len(ptxt)
                            else:
                                cur.append(ptxt)
                                total_len += len(ptxt)
                        if cur:
                            text = tidy_punct("\n\n".join(cur))
                            doc = {
                                "id": f"{parent_id}_c{chunk_idx:03d}",
                                "parent_id": parent_id,
                                "chunk_index": chunk_idx,
                                "article_id": arid_parent,
                                "journal": args.journal,
                                "journal_id": args.journal_id,
                                "title": display_title,
                                "raw_title": raw_title_out,
                                "author": author,
                                "section": section,
                                "volume": issue_volume,
                                "number": issue_number,
                                "date": date_final,
                                "year": year_final,
                                "page_no_start": page_no_start,
                                "page_no_end": page_no_end,
                                "page_label_start": page_label_start,
                                "page_label_end": page_label_end,
                                "text": text,
                            }
                            outfh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                            chunk_idx += 1

                        total_articles += 1
                        total_chunks += chunk_idx

                    print(f"[OK] {ddir}  chains={len(chains)}")

    print(f"\nIssues: {total_issues}  Articles: {total_articles}  Chunks: {total_chunks}")

if __name__ == "__main__":
    main()

