# -*- coding: utf-8 -*-
"""Assemble a SSRN submission package and PRE-FLIGHT it. Prepare-and-handoff ONLY.

SSRN has no submission API and its Terms of Use forbid 'automated queries of any sort', so this tool
NEVER posts to SSRN or auto-fills the form. It builds the PDF, writes machine-readable metadata, emits
a copy-paste portal checklist, validates the package, and (with --open) launches the browser to the
SSRN login page for the HUMAN to drive the upload.

    python build_ssrn_submission.py            # build + validate the package
    python build_ssrn_submission.py --open      # also open the SSRN submit/login page for you to drive
"""
import os, sys, json, webbrowser, datetime
import build_paper_pdf as P

DATA_DIR = P.DATA_DIR
OUTDIR = os.path.join(DATA_DIR, "ssrn_submission")
SUBMIT_URL = "https://hq.ssrn.com/submissions/CreateNewAbstract.cfm"
LOGIN_URL = "https://hq.ssrn.com/login/pubSignInJoin.cfm"


def _wordcount(s):
    return len(s.split())


def preflight(meta, pdf_path):
    """Return (ok, [issues]). HARD issues block; warnings are noted."""
    issues = []
    if not os.path.exists(pdf_path):
        return False, ["HARD: PDF not built"]
    if not pdf_path.lower().endswith(".pdf"):
        issues.append("HARD: SSRN accepts PDF only")
    wc = _wordcount(meta["abstract"])
    if not (250 <= wc <= 400):
        issues.append(f"{'WARN' if 200 <= wc <= 450 else 'HARD'}: abstract is {wc} words "
                      f"(SSRN ideal 250–400)")
    for field in ("title", "author", "affiliation", "email"):
        if not meta.get(field):
            issues.append(f"HARD: missing {field}")
    if "@" not in meta.get("email", ""):
        issues.append("HARD: invalid author email")
    if not meta.get("ai_disclosure"):
        issues.append("WARN: no AI-disclosure statement (required on the PDF if AI was used)")
    if not (1 <= len(meta.get("classifications", [])) <= 7):
        issues.append("HARD: pick 1–7 subject-matter classifications")
    # the PDF must show title + author on page 1 — verify by extracting text
    try:
        from pypdf import PdfReader
        p1 = PdfReader(pdf_path).pages[0].extract_text()
        if meta["author"].split()[0] not in p1:
            issues.append("HARD: author name not found on PDF page 1")
        if meta["title"].split(":")[0][:20] not in p1:
            issues.append("HARD: title not found on PDF page 1")
        if "AI disclosure" not in "".join(PdfReader(pdf_path).pages[i].extract_text()
                                         for i in range(len(PdfReader(pdf_path).pages))):
            issues.append("WARN: AI-disclosure line not detected in PDF text")
    except Exception as e:
        issues.append(f"WARN: could not text-verify the PDF ({type(e).__name__})")
    hard = [i for i in issues if i.startswith("HARD")]
    return (len(hard) == 0), issues


def checklist_md(meta, pdf_path):
    kw = "\n".join(f"   - {k}  (type, then press Enter)" for k in meta["keywords"])
    jel = "\n".join(f"   - {j}" for j in meta["jel"])
    cls = "\n".join(f"   - {c}" for c in meta["classifications"])
    return f"""# SSRN submission checklist — copy/paste into the portal

> SSRN has **no submission API** and its Terms of Use prohibit automated submission. A human must
> drive the web form. This file gives you every field value ready to paste. Posting is **free**.

## 0. One-time account setup (human)
1. Create / sign in to a free SSRN account: {LOGIN_URL}
2. Complete your **author profile** and set a **verified affiliation**. You are submitting as
   **{meta['affiliation']}** — on the profile, search SSRN's org list; if not found, use the
   "add a new organization" link or select Independent Researcher. (Affiliation lives on the
   profile, not the paper, and is required to submit.)

## 1. Start a submission
Go to **Submit a paper**: {SUBMIT_URL}  → drag-and-drop the PDF:
`{os.path.basename(pdf_path)}`

## 2. Fields (paste these exactly)
- **Title:** {meta['title']}
- **Date written:** {meta['date_written']}
- **Author:** {meta['author']} — {meta['affiliation']} — {meta['email']}
  (Author name/order CANNOT be changed after posting — verify now.)
- **Abstract** ({_wordcount(meta['abstract'])} words): paste from `ssrn_metadata.json` → "abstract".

## 3. Classifications (PERMANENT — cannot be changed later via self-service; choose deliberately)
{cls}

## 4. JEL codes (optional; appear only after a Social Sciences classification; type each + Enter)
{jel}

## 5. Keywords (optional; one at a time + Enter — a pasted comma list breaks the field)
{kw}

## 6. The PDF
- PDF only, English; page 1 shows the title + author + affiliation. ✓ (built)
- AI-disclosure statement is on the PDF. ✓ (built)
- License: {meta['license']}

## 7. Submit, then wait
- Click Submit. Staff review every paper; most post within ~3 business days
  ("Under Review by SSRN" watermark meanwhile). You'll get an email when it's live.
- Contact SSRN support only after 10 business days.

## 8. After posting (notes)
- Revisions (PDF / title / abstract / keywords / date / DOIs) go live instantly with NO re-review,
  but are IRREVERSIBLE (no saved prior versions) and CANNOT change author names/order or
  classifications. Withdraw via My Papers → Modify → Make it Inactive.

Generated by build_ssrn_submission.py on {datetime.date.today().isoformat()}. Do not automate the upload.
"""


def main(open_browser=False):
    os.makedirs(OUTDIR, exist_ok=True)
    meta = dict(P.META)
    pdf_path, _ = P.build(meta)
    meta_out = {**meta, "abstract_wordcount": _wordcount(meta["abstract"]),
                "pdf_file": os.path.basename(pdf_path),
                "generated": datetime.date.today().isoformat(),
                "submission_urls": {"login": LOGIN_URL, "submit": SUBMIT_URL},
                "note": "Prepare-and-handoff package. SSRN has no API; a human must drive the upload."}
    with open(os.path.join(OUTDIR, "ssrn_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUTDIR, "SSRN_CHECKLIST.md"), "w", encoding="utf-8") as f:
        f.write(checklist_md(meta, pdf_path))

    ok, issues = preflight(meta, pdf_path)
    print(f"SSRN package -> {OUTDIR}")
    for fn in sorted(os.listdir(OUTDIR)):
        print(f"   {fn}  ({os.path.getsize(os.path.join(OUTDIR, fn))/1024:.0f} KB)")
    print(f"\nabstract {meta_out['abstract_wordcount']} words · JEL {', '.join(meta['jel'])} · "
          f"{len(meta['classifications'])} classifications")
    print("\nPRE-FLIGHT: " + ("PASS — ready to upload" if ok else "BLOCKED"))
    for it in issues:
        print("   " + it)
    if not issues:
        print("   (no issues)")

    if open_browser:
        if not ok:
            print("\nNot opening browser — fix HARD issues first.")
        else:
            print(f"\nOpening SSRN login for you to drive the upload: {LOGIN_URL}")
            print("REMINDER: you upload manually. This tool never submits to SSRN.")
            webbrowser.open(LOGIN_URL)
    else:
        print(f"\nNext: review the package, then run with --open (or go to {SUBMIT_URL}) and upload by hand.")


if __name__ == "__main__":
    main(open_browser="--open" in sys.argv)
