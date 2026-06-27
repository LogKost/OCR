import re


def clean_mixed_text(text):
    if text == "Н/Д":
        return text

    text = re.sub(r"^\s*[\(\[\{]\s*", "C", text)
    text = re.sub(r"\s*[\)\]\}]$", "", text)

    words = text.split()
    fixed_words = []

    replacements = {
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "х": "x",
        "Т": "T",
    }

    for word in words:
        lat_count = sum(1 for c in word if "a" <= c <= "z" or "A" <= c <= "Z")
        cyr_count = sum(
            1 for c in word if "а" <= c <= "я" or "А" <= c <= "Я" or c in "ёЁ"
        )

        if lat_count > cyr_count and cyr_count > 0:
            for cyr_char, lat_char in replacements.items():
                word = word.replace(cyr_char, lat_char)
        fixed_words.append(word)

    text = " ".join(fixed_words)
    text = re.sub(r"\b[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    sticks_count = sum(1 for c in text if c in "Ii1l|![]()")
    total_chars = len(text.replace(" ", ""))

    if total_chars > 0 and (sticks_count / total_chars) > 0.6:
        return "Н/Д"

    return text if text else "Н/Д"
