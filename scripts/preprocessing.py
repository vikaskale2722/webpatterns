import os, json, re
import pandas as pd
from pathlib import Path
import spacy
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Load German spaCy model
nlp = spacy.load("de_core_news_sm", disable=["ner", "parser"])
STOPWORDS = nlp.Defaults.stop_words | {"gmbh", "kg", "mbh", "e.k."}

def clean_text(text):
    """Clean and lemmatize German text."""
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"\\b\\d{1,2}[./-]\\d{1,2}[./-]\\d{2,4}\\b", "", text)
    text = re.sub(r"[^\wäöüÄÖÜß\s]", " ", text)
    doc = nlp(text.lower())
    return " ".join([
        token.lemma_ for token in doc
        if token.is_alpha and token.lemma_ not in STOPWORDS
    ])

# Captcha detection patterns
captcha_patterns = [
    r"verify.*robot",
    r"captcha",
    r"please complete the security check",
    r"human verification",
    r"prove you are not a robot",
    r"cloudflare"
]

def is_captcha(text, file_name="", cid="", folder_path=""):
    """Return match info dict if likely CAPTCHA, else None."""
    lower_text = text.lower()
    if len(lower_text) < 1500:
        for pattern in captcha_patterns:
            if re.search(pattern, lower_text):
                return {
                    "cid": cid,
                    "file_name": file_name,
                    "matched_pattern": pattern,
                    "directory": str(folder_path)
                }
    return None

# Project-relative paths
project_root = Path(__file__).resolve().parents[1]
base_dir = project_root / "data/raw/company_cache/website-data"
output_csv = project_root / "data/cleaned/company_clean_docs.csv"
log_path = project_root / "data/logs/skipped_captcha_files.xlsx"

if not base_dir.exists():
    raise FileNotFoundError(f"Directory not found: {base_dir}")

# Processing loop
records = []
skipped_files_log = []
processed, skipped = 0, 0

for company_folder in base_dir.iterdir():
    if not company_folder.is_dir():
        continue

    try:
        with open(company_folder / "company_data.json", encoding="utf-8") as f:
            meta = json.load(f)
        cid = meta["crefonummer"]

        text_parts = []
        for txt_file in company_folder.glob("*.txt"):
            with open(txt_file, encoding="utf-8") as f:
                text = f.read()
                captcha_info = is_captcha(text, txt_file.name, cid, company_folder)
                if captcha_info:
                    logging.warning(f"Skipping CAPTCHA file: {txt_file.name} for {cid}")
                    skipped_files_log.append(captcha_info)
                    continue
                text_parts.append(text)

        if not text_parts:
            skipped += 1
            continue

        full_text = " ".join(text_parts)[:10000]
        cleaned_text = clean_text(full_text)

        records.append({
            "cid": cid,
            "name": meta.get("name", ""),
            "zweck": meta.get("zweck", ""),
            "wz_code": meta.get("code", ""),
            "address": meta.get("addr", ""),
            "web": meta.get("web", ""),
            "clean_text": cleaned_text
        })
        processed += 1

    except Exception as e:
        logging.error(f"[ERROR] {company_folder.name}: {e}")
        skipped += 1

# Save results
df = pd.DataFrame(records)
output_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_csv, index=False)

# Save skipped CAPTCHA log
if skipped_files_log:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(skipped_files_log).to_excel(log_path, index=False)
    logging.info(f"Logged {len(skipped_files_log)} skipped CAPTCHA files to: {log_path}")
else:
    logging.info("No CAPTCHA files were skipped.")

logging.info(f"Preprocessing complete: {processed} companies processed, {skipped} skipped.")
logging.info(f"Output saved to: {output_csv}")
