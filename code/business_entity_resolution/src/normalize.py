"""
High-performance normalization and token extraction for multi-lingual business entity resolution.
Handles US, India, France, and arbitrary open-set country data.
"""

import re
import unicodedata

# Common legal suffixes across US, India, UK, France, and International
LEGAL_PATTERNS = [
    # India / UK
    r'\b(pvt|private)\s*(ltd|limited)\b',
    r'\b(pvt|ltd|limited|llp|llc|inc|corp|corporation|company|co)\b',
    r'\b(enterprises?|solutions?|industries|services|technologies|associates?|group|holdings?)\b',
    # France
    r'\b(sarl|sasu|sas|s\.a\.s|s\.a|eurl|sci|snc|gie|scop|selarl)\b',
    r'\b(et\s+fils|et\s+freres|fils|freres|cie|societe)\b',
    # Common business terms
    r'\b(club|ecole|centre|center|pharmacie|distribution|commercial|internationale?)\b',
]
LEGAL_RE = re.compile(r'|'.join(LEGAL_PATTERNS), re.IGNORECASE)

# Common street abbreviations
STREET_ABBREVS = {
    'street': 'st', 'saint': 'st', 'avenue': 'ave', 'av': 'ave', 'road': 'rd', 'boulevard': 'blvd', 'bd': 'blvd',
    'drive': 'dr', 'lane': 'ln', 'place': 'pl', 'court': 'ct', 'highway': 'hwy', 'parkway': 'pkwy',
    'rue': 'r', 'allee': 'all', 'chemin': 'ch'
}

def strip_accents_and_clean(text: str) -> str:
    """Normalize unicode, strip accents for latin, clean punctuation."""
    if not text or text == "None" or text == "null":
        return ""
    # Normalize unicode NFKD and remove combining diacritics
    nfkd = unicodedata.normalize('NFKD', str(text))
    text = ''.join(c for c in nfkd if not unicodedata.combining(c))
    # Replace common separators with space
    text = re.sub(r'[\-_/\\,;:()\[\]{}"\'.<>+*&@#%^!=?|~`]', ' ', text)
    # Lowercase and collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def normalize_name(name: str) -> str:
    """Clean name, remove legal suffixes, normalize whitespace."""
    cleaned = strip_accents_and_clean(name)
    if not cleaned:
        return ""
    # Strip legal entity keywords
    core = LEGAL_RE.sub(' ', cleaned)
    core = re.sub(r'\s+', ' ', core).strip()
    return core if core else cleaned

def normalize_address(address: str) -> str:
    """Normalize address abbreviations and punctuation."""
    cleaned = strip_accents_and_clean(address)
    if not cleaned:
        return ""
    tokens = cleaned.split()
    norm_tokens = [STREET_ABBREVS.get(t, t) for t in tokens]
    return ' '.join(norm_tokens)

def extract_numbers(text: str) -> set:
    """Extract all standalone number tokens from text (house numbers, postal codes, phone numbers)."""
    if not text:
        return set()
    nums = re.findall(r'\b\d+\b', str(text))
    return set(nums)

def extract_acronym(text: str) -> str:
    """Extract acronym from words (e.g. 'International Business Machines' -> 'ibm')."""
    words = [w for w in text.split() if w]
    if len(words) >= 2:
        return ''.join(w[0] for w in words if w[0].isalnum())
    return ""

def get_core_tokens(text: str, min_len: int = 3, stopwords: set = None) -> set:
    """Extract significant discriminative tokens."""
    if not text:
        return set()
    tokens = text.split()
    res = set()
    for t in tokens:
        if len(t) >= min_len:
            if stopwords and t in stopwords:
                continue
            res.add(t)
    return res
