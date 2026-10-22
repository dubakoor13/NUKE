"""Adversarial checks for real-PDF chapter-index provenance algorithm.

Fixture text is SYNTHETIC, not a substitute for the actual Emrich PDF.
No nuclear rocket reactor calculations, data, formulas, or equations are
reproduced from the copyrighted source.
"""
from __future__ import annotations

from .emrich_reference_audit import (
    CHAPTERS, CRITICAL_CHAPTERS, chapter_matches_publisher_order,
    index_chapter_page_texts, normalized,
)


def reject(call, expected: str):
    try:
        call()
    except ValueError as e:
        assert expected in str(e), (expected, str(e))
    else:
        raise AssertionError("malformed chapter index incorrectly accepted")


def main() -> int:
    assert normalized("Neutron  BALANCE!\nEquation") == "neutron balance equation"
    pages=["Publisher front matter"]*420
    # Malicious TOC duplicate headings are NOT chapter body locations.
    pages[5]="Contents: "+", ".join(row[1] for row in CHAPTERS)
    # Even the second-edition printed page is not the SAME as the PDF page
    # index because the document includes front matter.
    selected={
        "5":(88,"5. Basic Nuclear Structure and Processes"),
        "6":(127,"Chapter 6\nNEUTRON FLUX ENERGY DISTRIBUTION"),
        "7":(137,"7\nNeutron Balance Equation and Transport Theory"),
        "8":(145,"CHAPTER 8\nMultigroup Neutron Diffusion Equations"),
        "9":(164,"9 Thermal Fluid Aspects of Nuclear Rockets"),
        "11":(218,"11 Nuclear Reactor Kinetics"),
        "13":(284,"13 Fuel Burnup and Transmutation"),
        "15":(319,"15 Materials for Nuclear Thermal Rockets"),
        "16":(338,"16 Nuclear Rocket Engine Testing"),
        "appendix_iii":(406,"Appendix III: Selected Data from NERVA Tests"),
    }
    for key,(page,text) in selected.items():
        pages[page-1]=text+"\n\nTextbook chapter starts here."
    results=index_chapter_page_texts(pages)
    for key,(page,_) in selected.items():
        assert results[key]==[page],(key,results[key])
    assert chapter_matches_publisher_order(results)
    assert all(results[str(n)] for n in CRITICAL_CHAPTERS)
    assert results["appendix_iii"]==[406]
    # A heading repeated halfway through chapter is an AMBIGUITY that
    # gets recorded as multiple candidates, never silently discarded.
    pages[187]="Introduction to neutron flux energy distribution"
    ambiguous=index_chapter_page_texts(pages)
    assert ambiguous["6"]==[127,188]
    assert chapter_matches_publisher_order(ambiguous)

    # Perturb the ordering to detect chapter mis-indexing.
    pages[319-1]="Different subsection"
    pages[120]="15 Materials for Nuclear Thermal Rockets"
    wrong=index_chapter_page_texts(pages)
    assert not chapter_matches_publisher_order(wrong)

    reject(lambda:index_chapter_page_texts(pages,min_front_matter_pages=-1),
           "minimum front matter")
    reject(lambda:index_chapter_page_texts(
        pages,min_front_matter_pages=True),"minimum front matter")

    print("Emrich 2023 PDF heading/page integrity regression: PASS (synthetic only)")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
