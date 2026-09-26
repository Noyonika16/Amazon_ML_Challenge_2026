import re
import unicodedata

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

STREET_ABBREVS = {
    'street': 'st', 'saint': 'st', 'avenue': 'ave', 'av': 'ave', 'road': 'rd', 'boulevard': 'blvd', 'bd': 'blvd',
    'drive': 'dr', 'lane': 'ln', 'place': 'pl', 'court': 'ct', 'highway': 'hwy', 'parkway': 'pkwy',
    'rue': 'r', 'allee': 'all', 'chemin': 'ch'
}

def strip_accents_and_clean(text: str) -> str:
    if not text or text == "None" or text == "null":
        return ""
    nfkd = unicodedata.normalize('NFKD', str(text))
    text = ''.join(c for c in nfkd if not unicodedata.combining(c))
    text = re.sub(r'[\-_/\\,;:()\[\]{}"\'.<>+*&@#%^!=?|~`]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def normalize_name(name: str) -> str:
    cleaned = strip_accents_and_clean(name)
    if not cleaned:
        return ""
    core = LEGAL_RE.sub(' ', cleaned)
    core = re.sub(r'\s+', ' ', core).strip()
    return core if core else cleaned

def normalize_address(address: str) -> str:
    cleaned = strip_accents_and_clean(address)
    if not cleaned:
        return ""
    tokens = cleaned.split()
    norm_tokens = [STREET_ABBREVS.get(t, t) for t in tokens]
    return ' '.join(norm_tokens)

def extract_numbers(text: str) -> set:
    if not text:
        return set()
    nums = re.findall(r'\b\d+\b', str(text))
    return set(nums)

def extract_acronym(text: str) -> str:
    words = [w for w in text.split() if w]
    if len(words) >= 2:
        return ''.join(w[0] for w in words if w[0].isalnum())
    return ""

def get_core_tokens(text: str, min_len: int = 3, stopwords: set = None) -> set:
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
