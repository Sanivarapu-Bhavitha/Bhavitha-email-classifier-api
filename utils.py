import re
import spacy

nlp = spacy.load("en_core_web_sm")

PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone_number": r"\b\d{10}\b",
    "aadhar_num": r"\b\d{4} \d{4} \d{4}\b",
    "credit_debit_no": r"\b(?:\d[ -]*?){13,16}\b",
    "cvv_no": r"\b\d{3}\b",
    "expiry_no": r"\b(0[1-9]|1[0-2])\/?([0-9]{2})\b",
    "dob": r"\b\d{2}/\d{2}/\d{4}\b"
}

def mask_text(email_body):
    masked_email = email_body
    entities = []

    for label, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, email_body):
            start, end = match.span()
            original = match.group()
            masked_email = masked_email.replace(original, f"[{label}]", 1)
            entities.append({
                "position": [start, end],
                "classification": label,
                "entity": original
            })

    doc = nlp(email_body)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            start, end = ent.start_char, ent.end_char
            original = ent.text
            masked_email = masked_email.replace(original, "[full_name]", 1)
            entities.append({
                "position": [start, end],
                "classification": "full_name",
                "entity": original
            })

    return masked_email, entities
