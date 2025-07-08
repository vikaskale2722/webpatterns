import os, json, re
import pandas as pd
from pathlib import Path
import spacy
import logging

#Version tag for traceability
PREPROCESS_VERSION = "v2.7-track-and-report"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

nlp = spacy.load("de_core_news_sm", disable=["ner", "parser"])
nlp.add_pipe("sentencizer")

EXTRA_STOPWORDS = {
    "gmbh", "kg", "mbh", "e.k.", "unternehmen", "service", "projekt", "kontakt", "angebot",
    "team", "partner", "mehr", "erfahren", "startseite", "impressum", "datenschutz", "agb",
    "cookie", "kunden", "kundenservice", "leistungen", "produkte", "lösungen",
    "daten", "verarbeitung", "personenbezogen", "personenbezogene", "einwilligung", "recht",
    "datenschutzerklärung", "hinweis", "dsgvo", "analytics", "cookies", "adresse", "mail",
    "art.", "abs.", "website", "websit", "google", "finden", "zweck", "uhr",
    "webseite", "telefon", "angebot", "e"
}
STOPWORDS = nlp.Defaults.stop_words | EXTRA_STOPWORDS

BOILERPLATE_PATTERNS = [
    r"\bstartseite\b", r"\bkontakt\b", r"\bimpressum\b", r"\bdatenschutz\b",
    r"\bagb\b", r"\bcookie\b", r"\bmenu\b", r"\bwillkommen\b", r"\bkarriere\b", r"\bteam\b",
    r"\bunternehmen\b", r"\bprodukte\b", r"\bleistungen\b", r"\bpartner\b", r"\bprojekte\b",
    r"\bdatenschutzerklärung\b", r"\bdatenschutz\b", r"\bpersonenbezogen\b", r"\bdsgvo\b",
    r"\beinwilligung\b", r"\bverarbeitung\b", r"\bhinweis\b", r"\bgoogle\b", r"\bmail\b",
    r"\baddress\b", r"\bwebsite\b", r"\bwebsit\b", r"\babs\b", r"\bart\b", r"\buhr\b",
    r"\bzweck\b", r"\bkunde\b"
]

GDPR_SECTION_PATTERNS = [
    r"datenschutzerklärung.*", r"hinweis.*daten", r"personenbezogen.*daten",
    r"gemäß.*dsgvo", r"ihre einwilligung.*", r"zur nutzung.*cookies", r"google analytics.*",
    r"alle rechte vorbehalten", r"verarbeitung.*daten"
]

FORMATTING_PATTERNS = [
    r"\be\b"
]

CAPTCHA_BLOCK_PATTERNS = [
    r"verify.*robot",
    r"captcha",
    r"please complete the security check",
    r"human verification",
    r"prove you are not a robot",
    r"cloudflare",
    r"bitte bestätigen.*mensch",
    r"sicherheitsüberprüfung",
    r"sicherheits-check",
    r"sie sind ein mensch",
    r"fehlermeldung",
    r"zugriff verweigert",
]

def strip_captcha_block(text):
    pattern = re.compile("|".join(CAPTCHA_BLOCK_PATTERNS), re.IGNORECASE)
    match = pattern.search(text)
    if match:
        if match.start() < 100:
            return ""
        return text[:match.start()]
    return text

def remove_boilerplate(text):
    for pat in GDPR_SECTION_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    for pat in FORMATTING_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return text

def clean_text(text):
    # Remove URLs & dates
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", "", text)
    text = remove_boilerplate(text)
    text = re.sub(r"[-–—−]+", " ", text)  # Remove all dash-like chars
    text = re.sub(r"[^\wäöüÄÖÜß\s\.]", " ", text)
    doc = nlp(text.lower())
    clean_sentences = []
    for sent in doc.sents:
        lemmas = [token.lemma_ for token in sent if token.is_alpha and token.lemma_ not in STOPWORDS]
        if lemmas:
            clean_sentences.append(" ".join(lemmas))
    result = ". ".join(clean_sentences)
    result = re.sub(r"[-–—−]+", " ", result)
    return result

#Paths
project_root = Path(__file__).resolve().parents[1]
base_dir = project_root / "data/raw/company_cache/website-data"
output_csv = project_root / "data/cleaned/company_clean_docs.csv"
log_path = project_root / "data/logs/skipped_captcha_files.xlsx"
summary_csv = project_root / "data/cleaned/preprocessing_summary.csv"
ignored_companies_csv = project_root / "data/cleaned/ignored_companies.csv"

if not base_dir.exists():
    raise FileNotFoundError(f"Directory not found: {base_dir}")

#Counters and logs
total_companies = 0            # total folders
total_txt_files = 0            # total .txt files
skipped_txt_files = 0          # .txt files skipped (CAPTCHA/junk)
companies_fully_skipped = 0    # no usable text
companies_small_text = 0       # too short after cleaning
ignored_companies = []         # details of all ignored companies

records = []
skipped_files_log = []
processed, skipped = 0, 0

#Main loop
for company_folder in base_dir.iterdir():
    if not company_folder.is_dir():
        continue
    total_companies += 1
    company_txt_count = 0
    company_txt_skipped = 0

    try:
        with open(company_folder / "company_data.json", encoding="utf-8") as f:
            meta = json.load(f)
        cid = meta["crefonummer"]
        cname = meta.get("name", "")

        text_parts = []
        for txt_file in company_folder.glob("*.txt"):
            company_txt_count += 1
            with open(txt_file, encoding="utf-8") as f:
                raw_text = f.read()
                stripped_text = strip_captcha_block(raw_text)
                if not stripped_text.strip():
                    company_txt_skipped += 1
                    skipped_txt_files += 1
                    skipped_files_log.append({
                        "cid": cid,
                        "file_name": txt_file.name,
                        "reason": "only CAPTCHA detected",
                        "directory": str(company_folder)
                    })
                    continue
                text_parts.append(stripped_text)

        total_txt_files += company_txt_count

        # If all .txt files were junk, log as fully skipped
        if not text_parts:
            companies_fully_skipped += 1
            skipped += 1
            ignored_companies.append({
                "cid": cid,
                "name": cname,
                "reason": "all files were junk/CAPTCHA"
            })
            continue

        full_text = ". ".join(text_parts)[:10000]
        full_text = re.sub(r"[-–—−]+", " ", full_text)
        cleaned_text = clean_text(full_text)
        cleaned_len = len(cleaned_text.split())

        # If cleaned text is too short, log as small text skip
        if cleaned_len < 30:
            companies_small_text += 1
            skipped += 1
            ignored_companies.append({
                "cid": cid,
                "name": cname,
                "reason": f"too short after cleaning ({cleaned_len} tokens)"
            })
            continue

        records.append({
            "cid": cid,
            "name": cname,
            "zweck": meta.get("zweck", ""),
            "wz_code": meta.get("code", ""),
            "address": meta.get("addr", ""),
            "web": meta.get("web", ""),
            "clean_text": cleaned_text,
            "preprocess_version": PREPROCESS_VERSION
        })
        processed += 1

    except Exception as e:
        logging.error(f"[ERROR] {company_folder.name}: {e}")
        skipped += 1
        ignored_companies.append({
            "cid": cid if 'cid' in locals() else "",
            "name": cname if 'cname' in locals() else company_folder.name,
            "reason": f"error: {str(e)}"
        })

#Save main outputs
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

# Save ignored companies details (all-junk and small text)
pd.DataFrame(ignored_companies).to_csv(ignored_companies_csv, index=False)

# Save summary CSV
summary_data = {
    "step": [
        "Total companies (raw)",
        "Total .txt files (raw)",
        "Total .txt files skipped (CAPTCHA/junk)",
        "Companies fully skipped (all junk/CAPTCHA)",
        "Companies skipped (too short after cleaning)",
        "Companies in cleaned CSV"
    ],
    "count": [
        total_companies,
        total_txt_files,
        skipped_txt_files,
        companies_fully_skipped,
        companies_small_text,
        len(df)
    ]
}
pd.DataFrame(summary_data).to_csv(summary_csv, index=False)

logging.info(f"Preprocessing complete: {processed} companies processed, {skipped} skipped.")
logging.info(f"Output saved to: {output_csv}")
logging.info(f"Summary saved to: {summary_csv}")
logging.info(f"Ignored company details saved to: {ignored_companies_csv}")
