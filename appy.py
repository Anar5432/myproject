# app.py (patched: robust answer-to-option mapping; no E-fallback overwrite)
import os
import re
import json
import random
import copy
import uuid
from pathlib import Path
import hashlib
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    session, send_file, send_from_directory
)
from werkzeug.security import generate_password_hash, check_password_hash
import fitz  # PyMuPDF
from werkzeug.utils import secure_filename
from pypdf import PdfReader

# --- Config ---
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATASET_DIR = BASE_DIR / "datasets"
EXAM_DIR = BASE_DIR / "exams"
ALLOWED_EXTENSIONS = {"pdf"}
ARCHIVE_FILE = (UPLOAD_DIR / "archive.json")
USERS_FILE = (BASE_DIR / "users.json")
USER_ARCHIVES_FILE = (BASE_DIR / "user_archives.json")
PAGE_IMG_DIR = UPLOAD_DIR / "pages"

UPLOAD_DIR.mkdir(exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True)
EXAM_DIR.mkdir(exist_ok=True)
PAGE_IMG_DIR.mkdir(exist_ok=True)
if not ARCHIVE_FILE.exists():
    with open(ARCHIVE_FILE, "w", encoding="utf-8") as _f:
        json.dump([], _f)
if not USERS_FILE.exists():
    with open(USERS_FILE, "w", encoding="utf-8") as _f:
        json.dump({}, _f)
if not USER_ARCHIVES_FILE.exists():
    with open(USER_ARCHIVES_FILE, "w", encoding="utf-8") as _f:
        json.dump({}, _f)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "replace-this-with-a-secure-random-key"

# --------------------------
# Helpers
# --------------------------
def normalize_text_for_compare(s):
    if s is None:
        return ""
    s = s.strip().lower()
    # remove common punctuation and extra spaces
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def compute_file_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def load_archive() -> list:
    try:
        with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []

def save_archive(entries: list) -> None:
    with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

def render_pdf_pages(pdf_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    zoom = 2.0  # ~144 DPI
    mat = fitz.Matrix(zoom, zoom)
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_file = out_dir / f"page_{i+1}.png"
        pix.save(str(out_file))
    doc.close()

def load_users() -> dict:
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(users: dict) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def load_user_archive(username: str) -> list:
    try:
        with open(USER_ARCHIVES_FILE, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    except Exception:
        mapping = {}
    return mapping.get(username, [])

def save_user_archive(username: str, entries: list) -> None:
    try:
        with open(USER_ARCHIVES_FILE, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    except Exception:
        mapping = {}
    mapping[username] = entries
    with open(USER_ARCHIVES_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

def current_username():
    return session.get("username")

# --------------------------
# PDF extraction & parsing (unchanged core but emits answer_letter if found)
# --------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    pdf = PdfReader(pdf_path)
    text = ""
    for idx, page in enumerate(pdf.pages, start=1):
        page_text = page.extract_text()
        text += f"[[PAGE {idx}]]\n"
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_pdf_robust(pdf_path: str) -> str:
    """
    Extract text from a PDF with a robust, math-friendly strategy:
    1) Try PyPDF native text extraction.
    2) If a page has too little text, try PyMuPDF block extraction (keeps layout better).
    3) If still too little, render the page and OCR it with Tesseract (if available),
       preserving inter-word spaces and math symbols as best as possible.
    Always emits page markers in the form of [[PAGE N]].
    """
    # Try to open via PyPDF and PyMuPDF
    try:
        pdf = PdfReader(pdf_path)
        num_pages = len(pdf.pages)
    except Exception:
        pdf = None
        num_pages = 0

    try:
        doc = fitz.open(str(pdf_path))
        num_pages_fitz = len(doc)
    except Exception:
        doc = None
        num_pages_fitz = 0

    # Check OCR availability lazily
    ocr_available = False
    pytesseract_mod = None
    Image_mod = None
    try:
        import pytesseract as _pyt
        from PIL import Image as _Image
        # ensure tesseract binary exists
        _ = _pyt.get_tesseract_version()
        pytesseract_mod = _pyt
        Image_mod = _Image
        ocr_available = True
    except Exception:
        ocr_available = False

    total_pages = max(num_pages, num_pages_fitz)
    out_parts = []

    def too_little(txt: str) -> bool:
        if txt is None:
            return True
        stripped = txt.strip()
        # Consider too little if fewer than 50 visible chars
        return len(stripped) < 50

    for i in range(total_pages):
        out_parts.append(f"[[PAGE {i+1}]]")
        page_text = ""

        # 1) Try PyPDF
        if pdf is not None and i < num_pages:
            try:
                p = pdf.pages[i]
                page_text = p.extract_text() or ""
            except Exception:
                page_text = ""

        # 2) Try PyMuPDF blocks if PyPDF yielded too little
        if too_little(page_text) and doc is not None and i < num_pages_fitz:
            try:
                page = doc.load_page(i)
                blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, ...)
                # sort top-to-bottom, then left-to-right
                blocks_sorted = sorted(
                    [b for b in blocks if len(b) >= 5 and b[4]],
                    key=lambda b: (round(b[1], 1), round(b[0], 1))
                )
                lines = []
                for b in blocks_sorted:
                    t = (b[4] or "").strip()
                    if t:
                        lines.append(t)
                page_text = "\n".join(lines)
            except Exception:
                pass

        # 3) OCR fallback if still too little
        if too_little(page_text) and doc is not None and i < num_pages_fitz and ocr_available:
            try:
                page = doc.load_page(i)
                # Use higher DPI for better OCR quality
                mat = fitz.Matrix(2.5, 2.5)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes("png")
                from io import BytesIO
                img = Image_mod.open(BytesIO(img_bytes))
                # Keep spaces and avoid aggressive segmentation
                config = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
                ocr_text = pytesseract_mod.image_to_string(img, config=config) or ""
                page_text = ocr_text
            except Exception:
                pass

        # light cleanup
        if not page_text:
            page_text = ""
        page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()
        out_parts.append(page_text)

    if doc is not None:
        try:
            doc.close()
        except Exception:
            pass

    return "\n".join(out_parts) + "\n"

def parse_mcqs(text: str):
    dataset = []
    question = None
    options = []
    answer = None
    answer_letter = None
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip() != ""]
    current_page = None

    inline_label_regex = re.compile(r"([A-E])[\.\)]\s*", re.IGNORECASE)

    def split_inline_options(line: str):
        """
        Split a line containing multiple inline labeled options like:
        "A) x   B) y   C) z" into [("A", "x"), ("B", "y"), ("C", "z")].
        Preserves math symbols and spacing within each option segment.
        """
        parts = []
        matches = list(inline_label_regex.finditer(line))
        if not matches:
            return parts
        for idx, m in enumerate(matches):
            label = m.group(1).upper()
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
            seg = line[start:end].strip()
            if seg.startswith("√") or seg.startswith("[x]") or seg.lower().startswith("(x)"):
                cleaned = re.sub(r"^[√\[\(x\)\]]+\s*", "", seg)
                parts.append((label, cleaned.strip(), True))
            else:
                parts.append((label, seg, False))
        return parts

    def is_mathy_option(text_line: str) -> bool:
        """Heuristic to detect short math-like option candidates."""
        s = text_line.strip()
        if len(s) == 0:
            return False
        if len(s) > 70:
            return False
        # Not a new question starter
        if re.match(r"^\d+[\.)]\s*", s):
            return False
        # Allow common math symbols, digits, letters for variables
        return re.match(r"^[\s\.,;:\-+*/=×÷•√^%‰°±≤≥<>⇒→←∑∏≈≃≅≡≠πeij\(\)\[\]\{\}0-9a-zA-Z,]+$", s) is not None

    for raw in lines:
        line = raw.strip()

        # Page marker
        m_page = re.match(r"^\[\[PAGE\s+(\d+)\]\]$", line)
        if m_page:
            current_page = int(m_page.group(1))
            continue

        # New question pattern
        m_q = re.match(r"^(\d+)[\.\)]\s*(.*)", line)
        if m_q:
            if question is not None:
                dataset.append({
                    "question": question.strip(),
                    "options": options,
                    "answer": answer,
                    "answer_letter": answer_letter,
                    "orig_page": current_page
                })
            question = m_q.group(2).strip()
            if question in {".", "..", "...", "-", "–", "—"}:
                question = ""
            options = []
            answer = None
            answer_letter = None
            continue

        # Option labelled A-E
        m_opt_label = re.match(r"^([A-E])[\.\)]\s*(.*)", line, re.IGNORECASE)
        if m_opt_label:
            # If multiple inline labels present, split them all from this line
            if len(list(inline_label_regex.finditer(line))) > 1:
                for lbl, seg, marked in split_inline_options(line):
                    options.append(seg)
                    if marked:
                        answer = seg
                        answer_letter = lbl
                continue
            # Single labeled option on this line
            label = m_opt_label.group(1).upper()
            opt_text = m_opt_label.group(2).strip()
            if opt_text.startswith("√") or opt_text.startswith("[x]") or opt_text.lower().startswith("(x)"):
                cleaned = re.sub(r"^[√\[\(x\)\]]+\s*", "", opt_text)
                options.append(cleaned.strip())
                answer = cleaned.strip()
                answer_letter = label
            else:
                options.append(opt_text)
            continue

        # Unlabelled bullets
        m_opt = re.match(r"^(?:\u2022\s*|[-–—]\s+)(.*)", line)
        if m_opt:
            opt_text = m_opt.group(1).strip()
            if opt_text.startswith("√") or opt_text.startswith("[x]") or opt_text.lower().startswith("(x)"):
                cleaned = re.sub(r"^[√\[\(x\)\]]+\s*", "", opt_text)
                options.append(cleaned.strip())
                answer = cleaned.strip()
            else:
                options.append(opt_text)
            continue

        # Math-style inline correct mark like "-1,4 √ -1,5"
        if question is not None and "√" in line and len(options) < 6:
            left, _, right = line.partition("√")
            left = left.strip()
            right = right.strip()
            added_any = False
            if left and is_mathy_option(left):
                options.append(left)
                added_any = True
            if right and is_mathy_option(right):
                options.append(right)
                answer = right
                added_any = True
            if added_any:
                continue

        # If looks like compact math options on one line, split by 2+ spaces
        if question is not None and is_mathy_option(line):
            parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
            if len(parts) > 1 and all(is_mathy_option(p) for p in parts):
                for p in parts:
                    options.append(p)
                continue

        # Math-style inline correct mark like "-1,4 √ -1,5"
        if question is not None and "√" in line and len(options) < 6:
            left, _, right = line.partition("√")
            left = left.strip()
            right = right.strip()
            added_any = False
            if left and is_mathy_option(left):
                options.append(left)
                added_any = True
            if right and is_mathy_option(right):
                options.append(right)
                answer = right
                added_any = True
            if added_any:
                continue

        # If looks like compact math options on one line, split by 2+ spaces
        if question is not None and is_mathy_option(line):
            parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
            if len(parts) > 1 and all(is_mathy_option(p) for p in parts):
                for p in parts:
                    options.append(p)
                continue

        # Explicit Answer: X
        if line.lower().startswith("answer:") or line.lower().startswith("correct:"):
            cleaned = re.sub(r"^answer[:\s]*|^correct[:\s]*", "", line, flags=re.IGNORECASE).strip()
            if re.match(r"^[A-E]$", cleaned, re.IGNORECASE) and options:
                letter = cleaned.upper()
                answer_letter = letter
                idx = ord(letter) - ord('A')
                if 0 <= idx < len(options):
                    answer = options[idx]
            else:
                answer = cleaned
            continue

        # Continuations
        if question is not None and options and (raw.startswith(" ") or raw.startswith("\t")):
            options[-1] = options[-1] + " " + line
            continue
        if question is not None and not options:
            if line not in {".", "..", "..."}:
                if question:
                    question += " " + line
                else:
                    question = line
            continue

        m_opt2 = re.match(r"^([A-E])\s+(.*)", line)
        if m_opt2:
            opt_text = m_opt2.group(2).strip()
            options.append(opt_text)
            continue

        if question is not None and options:
            options[-1] = options[-1] + " " + line
            continue

    if question is not None:
        dataset.append({
            "question": question.strip(),
            "options": options,
            "answer": answer,
            "answer_letter": answer_letter,
            "orig_page": current_page
        })

# Normalize strings and try to align answer text to an existing option
    for item in dataset:
        item["options"] = [o.strip() for o in item.get("options", []) if o and o.strip() != ""]
        if item.get("answer") and item["answer"] not in item["options"]:
            # robust search: normalized compare
            target_norm = normalize_text_for_compare(item["answer"])
            found = None
            for o in item["options"]:
                if normalize_text_for_compare(o) == target_norm:
                    found = o
                    break
            if found:
                item["answer"] = found
            # else keep answer text as-is but DO NOT mutate options (no forcing into E)
        # Fill empty question text if options exist (better UX for math PDFs)
        if (not item.get("question")) and item.get("options"):
            item["question"] = "Choose the correct answer."
    return dataset

def normalize_options(dataset, desired=5):
    out = copy.deepcopy(dataset)
    for item in out:
        opts = item.get("options", [])[:]
        ans = item.get("answer")
        ans_letter = item.get("answer_letter")
        opts = [o.strip() for o in opts if o is not None]

        # pad
        if len(opts) < desired:
            for i in range(len(opts)+1, desired+1):
                opts.append(f"Option {i} (auto-added)")

        # trim
        if len(opts) > desired:
            opts = opts[:desired]

        # if letter present and within bounds, align answer text to that option
        if ans_letter is not None:
            idx = ord(ans_letter.upper()) - ord('A')
            if idx < len(opts):
                item['answer'] = opts[idx]
            # else: out-of-range letter -> retain answer text (do not overwrite an option)

        # if no letter, try robust normalization to match option text (do not change options)
        if ans and ans not in opts:
            target_norm = normalize_text_for_compare(ans)
            matched = None
            for o in opts:
                if normalize_text_for_compare(o) == target_norm:
                    matched = o
                    break
            if matched:
                item['answer'] = matched
            # if still no match -> leave answer as-is (text), no forced insertion

        item['options'] = opts
    return out

def save_dataset_to_file(dataset, base_filename: str):
    for idx, item in enumerate(dataset, start=1):
        if isinstance(item, dict) and item.get("orig_index") is None:
            item["orig_index"] = idx

    safe = secure_filename(base_filename)
    out_path = DATASET_DIR / f"{Path(safe).stem}_dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    return str(out_path)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# --------------------------
# Exam persistence helpers (unchanged)
# --------------------------
def make_exam_filename():
    return f"exam_{uuid.uuid4().hex}.json"

def save_exam_state(exam_questions, answers):
    fname = make_exam_filename()
    path = EXAM_DIR / fname
    payload = {
        "questions": exam_questions,
        "answers": answers
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return fname

def load_exam_state(fname):
    if not fname:
        return None, None
    path = EXAM_DIR / fname
    if not path.exists():
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("questions"), payload.get("answers")

def update_exam_answers(fname, answers):
    path = EXAM_DIR / fname
    if not path.exists():
        return False
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["answers"] = answers
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return True

def remove_exam_file(fname):
    if not fname:
        return
    path = EXAM_DIR / fname
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

# --------------------------
# Routes (small adjustments in result mapping)
# --------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    dataset_file = session.get("dataset_file")
    available = 0
    if not current_username():
        # If not authenticated, go to login first
        return redirect(url_for("login"))
    if dataset_file and os.path.exists(dataset_file):
        with open(dataset_file, "r", encoding="utf-8") as f:
            ds = json.load(f)
            available = len(ds)

    if request.method == "POST":
        try:
            min_q = int(request.form.get("min_question", 1))
        except ValueError:
            min_q = 1
        try:
            max_q_raw = request.form.get("max_question", "").strip()
            max_q = int(max_q_raw) if max_q_raw else available
        except ValueError:
            max_q = available

        try:
            num_q = int(request.form.get("num_questions", 10))
        except ValueError:
            num_q = 10

        if not dataset_file or not os.path.exists(dataset_file):
            flash("No dataset uploaded yet. Please upload a PDF first.", "error")
            return redirect(url_for("index"))

        with open(dataset_file, "r", encoding="utf-8") as f:
            ds = json.load(f)

        min_q = max(1, min_q)
        max_q = min(len(ds), max_q)
        if min_q > max_q:
            flash("Invalid range: 'from' is greater than 'to'.", "error")
            return redirect(url_for("index"))

        pool = ds[(min_q - 1):max_q]
        if num_q > len(pool):
            flash(f"You requested {num_q} questions but only {len(pool)} are available in the chosen range.", "info")
            num_q = len(pool)

        random.shuffle(pool)
        exam_questions = pool[:num_q]

        answers = [None] * len(exam_questions)
        fname = save_exam_state(exam_questions, answers)
        session["exam_file"] = fname
        session.modified = True

        return redirect(url_for("exam"))

    archive = load_user_archive(current_username()) if current_username() else []
    current_pdf_filename = session.get("current_pdf_filename")
    current_pdf_url = None
    if current_pdf_filename:
        current_pdf_url = url_for("serve_upload", filename=current_pdf_filename)
    return render_template("index.html", page="home", available=available, archive=archive, current_pdf_url=current_pdf_url, username=current_username())

@app.route("/upload", methods=["POST"])
def upload():
    if "pdf_file" not in request.files:
        flash("No file part in the request.", "error")
        return redirect(url_for("index"))

    file = request.files["pdf_file"]
    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = UPLOAD_DIR / filename
        file.save(save_path)

        file_hash = compute_file_sha256(save_path)
        main_archive = load_archive()
        existing = next((e for e in main_archive if e.get("hash") == file_hash), None)

        if existing:
            if existing.get("stored_filename") != filename and save_path.exists():
                try:
                    save_path.unlink()
                except Exception:
                    pass
            ds_path = existing.get("dataset_path")
            if ds_path and os.path.exists(ds_path):
                session["dataset_file"] = ds_path
                session["current_pdf_filename"] = existing.get("stored_filename")
                session.modified = True
                # add to user archive if logged in
                if current_username():
                    uarc = load_user_archive(current_username())
                    if not any(x.get("hash") == file_hash for x in uarc):
                        uarc.append({
                            "hash": file_hash,
                            "original_filename": existing.get("original_filename", filename),
                            "stored_filename": existing.get("stored_filename", filename),
                            "dataset_path": ds_path,
                            "num_questions": existing.get("num_questions")
                        })
                        save_user_archive(current_username(), uarc)
                flash("This PDF is already in the archive. Reusing its questions.", "info")
                return redirect(url_for("index"))

        try:
            # Robust extraction (handles math by combining PyPDF, PyMuPDF and OCR fallback)
            try:
                text = extract_text_from_pdf_robust(str(save_path))
            except Exception:
                # fallback to basic extractor
                text = extract_text_from_pdf(str(save_path))
            dataset = parse_mcqs(text)
            if not dataset:
                flash("No questions detected in the PDF. Try a different file or check formatting.", "error")
                return redirect(url_for("index"))

            normalized = normalize_options(dataset, desired=5)

            padded = sum(1 for a, b in zip(dataset, normalized) if len(a.get("options", [])) < len(b.get("options", [])))
            trimmed = sum(1 for a, b in zip(dataset, normalized) if len(a.get("options", [])) > len(b.get("options", [])))
            answer_fixed = sum(1 for a, b in zip(dataset, normalized) if a.get("answer") and a.get("answer") != b.get("answer"))

            ds_path = save_dataset_to_file(normalized, filename)
            # Render per-page images for visuals (math diagrams, etc.)
            try:
                render_pdf_pages(save_path, PAGE_IMG_DIR / file_hash)
            except Exception:
                pass

            session["dataset_file"] = ds_path
            session["current_pdf_filename"] = filename
            session.modified = True

            # persist in main archive
            if not existing:
                main_archive.append({
                    "hash": file_hash,
                    "original_filename": filename,
                    "stored_filename": filename,
                    "dataset_path": ds_path,
                    "num_questions": len(normalized),
                    "page_image_dir": str((PAGE_IMG_DIR / file_hash).relative_to(BASE_DIR))
                })
                save_archive(main_archive)
            # persist in user archive
            if current_username():
                uarc = load_user_archive(current_username())
                if not any(x.get("hash") == file_hash for x in uarc):
                    uarc.append({
                        "hash": file_hash,
                        "original_filename": filename,
                        "stored_filename": filename,
                        "dataset_path": ds_path,
                        "num_questions": len(normalized)
                    })
                    save_user_archive(current_username(), uarc)

            msg = f"Uploaded and parsed {len(normalized)} questions successfully."
            extras = []
            if padded:
                extras.append(f"{padded} question(s) padded to 5 options")
            if trimmed:
                extras.append(f"{trimmed} question(s) trimmed to 5 options")
            if answer_fixed:
                extras.append(f"{answer_fixed} answer(s) normalized")
            if extras:
                msg += " (" + "; ".join(extras) + ")"

            flash(msg, "success")
        except Exception as e:
            flash(f"Error processing PDF: {e}", "error")
            return redirect(url_for("index"))
    else:
        flash("Invalid file type. Please upload a PDF.", "error")

    return redirect(url_for("index"))

@app.route("/exam", methods=["GET", "POST"])
def exam():
    fname = session.get("exam_file")
    exam_questions, answers = load_exam_state(fname)
    if exam_questions is None or answers is None:
        flash("No active exam. Please start an exam from the home screen.", "error")
        return redirect(url_for("index"))

    q_index = request.args.get("q_index")
    if q_index is None:
        q_index = 0
    else:
        try:
            q_index = int(q_index)
        except ValueError:
            q_index = 0

    if request.method == "POST":
        selected = request.form.get("answer")
        posted_index = request.form.get("q_index")
        try:
            posted_index = int(posted_index)
        except (TypeError, ValueError):
            posted_index = q_index
        if 0 <= posted_index < len(answers):
            answers[posted_index] = selected
            update_exam_answers(fname, answers)

        if posted_index + 1 >= len(exam_questions):
            return redirect(url_for("result"))
        else:
            return redirect(url_for("exam", q_index=posted_index + 1))

    total = len(exam_questions)
    if q_index < 0:
        q_index = 0
    if q_index >= total:
        q_index = total - 1

    # Prepare question for display (shuffle options)
    question = copy.deepcopy(exam_questions[q_index])
    original_options = question.get("options", [])[:]
    shuffled_options = original_options[:]
    random.shuffle(shuffled_options)
    question["display_options"] = shuffled_options
    if question.get("answer") in shuffled_options:
        question["correct_index_in_display"] = shuffled_options.index(question.get("answer"))
    else:
        question["correct_index_in_display"] = None

    current_answer = answers[q_index] if q_index < len(answers) else None

    # try to derive a likely PDF page for the current question
    page_hint = None
    if isinstance(exam_questions[q_index], dict):
        page_hint = exam_questions[q_index].get("orig_page")

    return render_template(
        "index.html",
        page="exam",
        question=question,
        q_index=q_index,
        total=total,
        current_answer=current_answer,
        current_pdf_url=url_for("serve_upload", filename=session.get("current_pdf_filename")) if session.get("current_pdf_filename") else None,
        current_pdf_page=page_hint
    )

@app.route("/result", methods=["GET"])
def result():
    fname = session.get("exam_file")
    exam_questions, answers = load_exam_state(fname)
    if not exam_questions or answers is None:
        flash("No exam data found.", "error")
        return redirect(url_for("index"))

    total = len(exam_questions)
    score = 0
    detailed = []
    status_list = []
    for q, your in zip(exam_questions, answers):
        correct_text = q.get("answer")
        # Prefer letter mapping if present
        correct_index = None
        if q.get("answer_letter"):
            idx = ord(q.get("answer_letter").upper()) - ord('A')
            if 0 <= idx < len(q.get("options", [])):
                correct_index = idx
        else:
            # try robust normalized text matching
            if correct_text:
                target = normalize_text_for_compare(correct_text)
                for i, o in enumerate(q.get("options", [])):
                    if normalize_text_for_compare(o) == target:
                        correct_index = i
                        break
        if your == correct_text:
            status = "correct"
            score += 1
        elif your is None:
            status = "unanswered"
        else:
            status = "incorrect"
        status_list.append(status)
        detailed.append({
            "question": q.get("question"),
            "options": q.get("options"),
            "orig_index": q.get("orig_index"),
            "correct_answer": correct_text,
            "correct_index": correct_index,   # None if not mappable
            "your_answer": your,
            "status": status
        })

    return render_template(
        "index.html",
        page="result",
        total=total,
        score=score,
        status_list=status_list,
        detailed=detailed,
        letters=[chr(ord('A')+i) for i in range(26)]
    )

@app.route("/download_results", methods=["GET"])
def download_results():
    fname = session.get("exam_file")
    exam_questions, answers = load_exam_state(fname)
    if not exam_questions or answers is None:
        flash("No exam data found.", "error")
        return redirect(url_for("index"))

    out = []
    for idx, (q, your) in enumerate(zip(exam_questions, answers), start=1):
        out.append({
            "index": idx,
            "question": q.get("question"),
            "options": q.get("options"),
            "correct_answer": q.get("answer"),
            "your_answer": your
        })

    tmp_path = DATASET_DIR / "last_results.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return send_file(tmp_path, as_attachment=True, download_name="results.json")

@app.route("/reset", methods=["GET"])
def reset():
    fname = session.pop("exam_file", None)
    remove_exam_file(fname)
    session.pop("answers", None)
    flash("Exam reset. You can start again.", "info")
    return redirect(url_for("index"))

@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/select_pdf", methods=["GET"])
def select_pdf():
    file_hash = request.args.get("hash")
    entry = None
    # first search in user archive
    if current_username():
        uarc = load_user_archive(current_username())
        entry = next((e for e in uarc if e.get("hash") == file_hash), None)
    # fallback to main archive
    if entry is None:
        archive = load_archive()
        entry = next((e for e in archive if e.get("hash") == file_hash), None)
    if not entry:
        flash("PDF not found in archive.", "error")
        return redirect(url_for("index"))

    ds_path = entry.get("dataset_path")
    if not ds_path or not os.path.exists(ds_path):
        flash("Dataset for this PDF is missing.", "error")
        return redirect(url_for("index"))

    session["dataset_file"] = ds_path
    session["current_pdf_filename"] = entry.get("stored_filename")
    session.modified = True
    flash("Selected archived PDF.", "success")
    return redirect(url_for("index"))

@app.route("/account", methods=["GET"])
def account():
    if not current_username():
        flash("Please log in to view your account.", "error")
        return redirect(url_for("login"))
    users = load_users()
    user = users.get(current_username()) or {}
    info = {
        "username": current_username(),
        "email": user.get("email"),
        "full_name": user.get("full_name")
    }
    return render_template("index.html", page="account", account=info, username=current_username())

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        users = load_users()
        # allow login via username or email
        user = users.get(username)
        if not user and "@" in username:
            for uname, u in users.items():
                if u.get("email", "").strip().lower() == username.lower():
                    user = u
                    username = uname
                    break
        if not user or not check_password_hash(user.get("password_hash", ""), password):
            flash("Invalid credentials.", "error")
            return redirect(url_for("login"))
        session["username"] = username
        flash("Logged in successfully.", "success")
        return redirect(url_for("index"))
    return render_template("index.html", page="login", username=current_username())

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        email = request.form.get("email", "").strip()
        full_name = request.form.get("full_name", "").strip()
        if not username or not password or not email:
            flash("Username and password required.", "error")
            return redirect(url_for("signup"))
        users = load_users()
        if username in users:
            flash("Username already exists.", "error")
            return redirect(url_for("signup"))
        # email uniqueness
        for u in users.values():
            if u.get("email", "").strip().lower() == email.lower():
                flash("Email already in use.", "error")
                return redirect(url_for("signup"))
        users[username] = {
            "password_hash": generate_password_hash(password),
            "email": email,
            "full_name": full_name
        }
        save_users(users)
        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("index.html", page="signup", username=current_username())

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("username", None)
    flash("Logged out.", "info")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
