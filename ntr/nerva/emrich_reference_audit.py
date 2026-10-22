"""Source-audit Emrich 2023 PDF already checked into the NUKE repository.

Produces ONLY chapter/page metadata and a core-physics validation crosswalk:
no copied textbook pages, formulas, exercises, tables, or test datasets.
Actual contents must be read from the repository's own 2023 PDF; never
substitute a 2016 first-edition page number or an online generic synopsis.

This is documentary PROVENANCE, not physical NASA NERVA validation or
evidence for a specific reactor geometry, nuclear data library or fuel load.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

PDF_FILENAME = (
    "William J. Emrich Jr. - Principles of Nuclear Rocket Propulsion-"
    "Butterworth-Heinemann (2023).pdf"
)

# The *printed* opening page numbers below are from the public Google Books
# 2023 SECOND-edition table of contents, NOT PDF page indices. The latter
# must be observed by opening the actual repository PDF and are reported
# separately. No claim of verified equation/page body text is made.
CHAPTERS = (
    (5, "Basic nuclear structure and processes", 59,
     ("ntr/nerva/audit_nuclear_data_temperatures.py",
      "ntr/nerva/executed_material_kerma_qualification.py"),
     "Cross-section identity, isotope provenance, heating-data coverage"),
    (6, "Neutron flux energy distribution", 97,
     ("ntr/nerva/neutron_energy_groups.py",
      "ntr/nerva/source_energy_audit.py",
      "ntr/nerva/source_phase_space_audit.py"),
     "Incoming neutron energy groups vs observed source and material flux"),
    (7, "Neutron balance equation and transport theory", 107,
     ("ntr/nerva/core_neutron_balance.py",
      "ntr/nerva/core_neutron_leakage_history.py",
      "ntr/nerva/core_neutron_economy_validation.py"),
     "Neutron balance, leakage, multiplicative scattering and source banking"),
    (8, "Multigroup neutron diffusion equations", 115,
     ("ntr/nerva/source_resolution_audit.py",
      "ntr/nerva/core_criticality_physical_qualification.py"),
     "Diffusion is a lower-fidelity independent comparison ONLY when "
     "condensed multigroup data and boundary assumptions are justified"),
    (9, "Thermal fluid aspects of nuclear rockets", 135,
     ("ntr/nerva/heating_data_qualification.py",
      "ntr/nerva/thermal.py"),
     "Neutronic deposited-energy source is currently UNQUALIFIED"),
    (11, "Nuclear reactor kinetics", 187,
     ("ntr/nerva/openmc_transient.py",),
     "Defer kinetics qualification until criticality/source and delayed "
     "neutron nuclear data are physically established"),
    (13, "Fuel burnup and transmutation", 251,
     ("ntr/nerva/model.py",),
     "Depletion validation not established for executed cold surrogate"),
    (15, "Materials for nuclear thermal rockets", 287,
     ("ntr/nerva/core_material_temperature_qualification.py",
      "ntr/nerva/physical_scattering_pair_qualification.py",
      "ntr/nerva/executed_material_kerma_qualification.py"),
     "Hot material nuclear data and bound scattering remain unqualified"),
    (16, "Nuclear rocket engine testing", 305,
     ("ntr/nerva/physical_core_reference_validation.py",
      "ntr/nerva/NASA_FIDELITY_MATRIX.md"),
     "Do NOT mix historical test points with model predictions"),
    (None, "Selected data from NERVA tests", None,
     ("ntr/nerva/physical_core_reference_validation.py",),
     "Appendix III is a secondary compilation; independent NASA test "
     "documents are needed for any benchmark claim"),
)

MIN_TEXT_SEARCHABLE_PAGES = 20
CRITICAL_CHAPTERS = frozenset((5, 6, 7, 8, 15))


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def index_chapter_page_texts(
    page_texts: Iterable[str], *, min_front_matter_pages: int = 20
) -> dict[str, list[int]]:
    """Find chapter-title occurrences near the TOP of actual PDF pages.

    Return all near-heading candidates to expose duplicated section headings
    rather than silently choosing a desirable page. Ignore early TOC hits.
    """
    if (isinstance(min_front_matter_pages, bool)
            or not isinstance(min_front_matter_pages, int)
            or min_front_matter_pages < 0):
        raise ValueError("invalid minimum front matter length")
    out = {str(num) if num is not None else "appendix_iii": []
           for num,_,_,_,_ in CHAPTERS}
    for idx, text in enumerate(page_texts):
        if idx < min_front_matter_pages:
            continue
        head = normalized(text[:1600])
        for number, title, _, _, _ in CHAPTERS:
            needle = normalized(title)
            # Headings may contain a separate CHAPTER/numeral line. A title
            # deep within later explanatory paragraphs is NOT an opening.
            if needle in head[:950]:
                out[str(number) if number is not None else "appendix_iii"].append(idx+1)
    return out


def chapter_matches_publisher_order(chapter_pages: dict[str, list[int]]) -> bool:
    ordered = []
    for number,_,_,_,_ in CHAPTERS:
        if number is None:
            continue
        pages = chapter_pages.get(str(number), [])
        if pages:
            ordered.append((number,pages[0]))
    return all(a[1] < b[1] for a,b in zip(ordered,ordered[1:]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda:src.read(1<<20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_real_emrich_pdf(path: Path, *, repo_root: Path) -> dict:
    path = Path(path)
    repo_root = Path(repo_root)
    if not path.is_file() or path.name != PDF_FILENAME:
        raise ValueError("canonical repository Emrich 2023 PDF not found")
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError("not the actual repository PDF file")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("install pypdf for actual source-PDF inspection") from exc
    reader = PdfReader(str(path), strict=True)
    if reader.is_encrypted or len(reader.pages) < 320:
        raise ValueError("actual Emrich second-edition PDF is encrypted or incomplete")
    all_texts = []
    n_text = 0
    for page in reader.pages:
        page_text = page.extract_text(extraction_mode="plain") or ""
        if normalized(page_text):
            n_text += 1
        all_texts.append(page_text)
    if n_text < MIN_TEXT_SEARCHABLE_PAGES:
        raise ValueError("PDF text not extractable: cannot verify actual page headings")
    front = normalized(" ".join(all_texts[:25]))
    if "principles of nuclear rocket propulsion" not in front or "emrich" not in front:
        raise ValueError("wrong actual source PDF title/author (not Emrich)")
    found = index_chapter_page_texts(all_texts)
    found_critical = [
        n for n in CRITICAL_CHAPTERS if len(found[str(n)]) > 0
    ]
    if len(found_critical) < 4:
        raise ValueError(
            "insufficient Emrich 2023 nuclear physics chapter headings in real PDF: "
            + repr(found)
        )
    if not chapter_matches_publisher_order(found):
        raise ValueError(
            "PDF chapter headings out of order or conflated with front matter"
        )
    chapters = []
    for num, title, printed, linked, purpose in CHAPTERS:
        key = str(num) if num is not None else "appendix_iii"
        pages = found[key]
        candidates = [p for p in pages if p>=21]
        chapter = {
            "chapter":num if num is not None else "Appendix III",
            "chapter_heading":title,
            "publisher_second_edition_printed_start_page":printed,
            "real_repository_pdf_page_candidates_1_based":candidates,
            "first_pdf_page_1_based_if_detected":candidates[0] if candidates else None,
            "page_heading_verified_in_actual_pdf":bool(candidates),
            "existing_nuke_files_to_review":list(linked),
            "scientific_validation_use":purpose,
            "chapter_content_and_equations_critically_reviewed":False,
            "historical_reactor_benchmark_validated":False,
        }
        for file in linked:
            if not (repo_root / file).is_file():
                raise ValueError("NUKE source file referenced in crosswalk missing: "+file)
        chapters.append(chapter)
    return {
        "audit":"EXACT_REPOSITORY_EMRICH_2023_SECOND_EDITION_PDF_CHAPTER_PROVENANCE",
        "pdf_repository_path":path.name,
        "source_pdf_sha256":_sha256(path),
        "source_pdf_bytes":path.stat().st_size,
        "source_pdf_page_count":len(reader.pages),
        "pages_with_extractable_text":n_text,
        "edition":"Second edition, Elsevier / Butterworth-Heinemann (2023)",
        "printed_start_pages_provenance":"Google Books 2023 2nd-edition table of contents",
        "publisher_toc_url":"https://books.google.com/books?id=haCSEAAAQBAJ",
        "published_edition_verification_url":
            "https://www.educate.elsevier.com/book/details/9780323900300",
        "critical_chapter_headings_found":sorted(found_critical),
        "critical_chapter_headings_required":sorted(CRITICAL_CHAPTERS),
        "chapter_headings_in_increasing_pdf_order":True,
        "chapter_location_crosswalk":chapters,
        "all_referenced_nuke_paths_exist":True,
        "nuke_calculations_benchmarked_against_emrich_exercises":False,
        "nasa_nerva_test_data_validated_against_primary_sources":False,
        "actual_pdf_equations_transcribed_and_independently_checked":False,
        "current_nerva_thermal_neutronic_physics_qualified":False,
        "interpretation":(
            "A traceable source-document index, NOT an equation extraction "
            "or validation certificate. Publisher printed page numbers and "
            "repository PDF page indices are intentionally distinct. A "
            "textbook's sample calculations and Appendix III compiled "
            "NERVA data are not by themselves primary NASA reactor "
            "measurements. No copyrighted passages, figures or tables are "
            "reproduced. No reactor physics or operating point changed."
        ),
    }


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf",type=Path,default=Path(PDF_FILENAME))
    ap.add_argument("--repo-root",type=Path,default=Path("."))
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    report=audit_real_emrich_pdf(args.pdf,repo_root=args.repo_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print("Verified actual Emrich source SHA256:",report["source_pdf_sha256"])
    print("Actual source PDF pages:",report["source_pdf_page_count"])
    for row in report["chapter_location_crosswalk"]:
        print(
            "Emrich",row["chapter"],"printed",
            row["publisher_second_edition_printed_start_page"],
            "actual PDF page",row["first_pdf_page_1_based_if_detected"]
        )
    print("NERVA historical test-point validation: NOT ESTABLISHED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
