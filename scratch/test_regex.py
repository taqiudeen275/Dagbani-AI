import re
import sys
import unicodedata

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

class DagbaniLinguisticNormalizer:
    ASCII_TO_BGL_LEXICON = {
        "nyela": "nyɛla", "be": "bɛ", "sheli": "shɛli", "sheba": "shɛba",
        "vieli": "viɛli", "vielim": "viɛlim", "paga": "paɣa", "pagaba": "paɣaba",
        "zhema": "ʒɛma", "zhem": "ʒɛm", "ngmani": "ŋmani", "ngma": "ŋma",
        "ngmande": "ŋmande", "kpe": "kpɛ", "zon": "zɔŋ", "wom": "wʊm",
        "bogsi": "bɔɣisi", "kpalinzhoo": "kpalinʒoo", "kpalingzhoo": "kpalinʒoo",
        "karimzon": "karimzɔŋ", "bihi": "bihi", "pukpara": "pukpara", "pukpariba": "pukpariba"
    }
    
    def normalize_text(self, text: str, for_asr: bool = True) -> str:
        if not text:
            return ""
        norm = unicodedata.normalize("NFC", str(text).strip())
        norm = re.sub(r"[‘’`′´]", "'", norm)
        norm = re.sub(r'["“”«»]', '', norm)
        
        tokens = norm.split()
        converted = []
        for tok in tokens:
            clean_tok = re.sub(r"[^\w'ɛɔɣŋʒƐƆƔŊƷ-]", "", tok)
            low = clean_tok.lower()
            if low in self.ASCII_TO_BGL_LEXICON:
                bgl = self.ASCII_TO_BGL_LEXICON[low]
                if clean_tok.isupper():
                    bgl = bgl.upper()
                elif clean_tok and clean_tok[0].isupper():
                    bgl = bgl.capitalize()
                converted.append(tok.replace(clean_tok, bgl))
            else:
                converted.append(tok)
        norm = " ".join(converted)
        
        if for_asr:
            norm = norm.lower()
            # Replace all non-alphanumeric/non-apostrophe/non-hyphen chars with space
            norm = re.sub(r"[^\w\s'-]", " ", norm, flags=re.UNICODE)
            # Remove isolated apostrophes
            norm = re.sub(r"\s+'|'\s+|^'|'$", " ", norm)
            # Collapse whitespace
            norm = re.sub(r"\s+", " ", norm).strip()
        return norm

normalizer = DagbaniLinguisticNormalizer()
test_samples = [
    ("Nyela paga mini bihi ban be Tamale.", "nyɛla paɣa mini bihi ban bɛ tamale"),
    ("“Dagbaŋ kaya ni ta'ada nyɛla din mali yaa pam!”", "dagbaŋ kaya ni ta'ada nyɛla din mali yaa pam"),
    ("O biɛla Yendi zúŋɔ ka nyɛ pukpara.", "o biɛla yendi zúŋɔ ka nyɛ pukpara"),
    ("Ti kpalinzhoo maa nyela din vieli pam.", "ti kpalinʒoo maa nyɛla din viɛli pam"),
]

for r, exp in test_samples:
    res = normalizer.normalize_text(r, for_asr=True)
    status = "PASS" if res == exp else "FAIL"
    print(f"[{status}] In: '{r}' -> Out: '{res}' (Expected: '{exp}')")
