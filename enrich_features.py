import pandas as pd

def get_cv_reference():
    cv_reference = {
        "paa": {"place_simplified": "labial", "place": "bilabial", "manner": "stop", "voicing": "voiceless"},
        "waa": {"place_simplified": "labial", "place": "bilabial", "manner": "approximant", "voicing": "voiced"},
        "shaa": {"place_simplified": "coronal", "place": "postalveolar", "manner": "fricative", "voicing": "voiceless"},
        "haa": {"place_simplified": "laryngeal", "place": "glottal", "manner": "fricative", "voicing": "voiceless"},
        "taa": {"place_simplified": "coronal", "place": "alveolar", "manner": "stop", "voicing": "voiceless"},
        "maa": {"place_simplified": "labial", "place": "bilabial", "manner": "nasal", "voicing": "voiced"},
        "vaa": {"place_simplified": "labial", "place": "labiodental", "manner": "fricative", "voicing": "voiced"},
        "daa": {"place_simplified": "coronal", "place": "alveolar", "manner": "stop", "voicing": "voiced"},
        "kaa": {"place_simplified": "dorsal", "place": "velar", "manner": "stop", "voicing": "voiceless"},
        "thaa": {"place_simplified": "coronal", "place": "dental", "manner": "fricative", "voicing": "voiceless"},
        "gaa": {"place_simplified": "dorsal", "place": "velar", "manner": "stop", "voicing": "voiced"},
        "saa": {"place_simplified": "coronal", "place": "alveolar", "manner": "fricative", "voicing": "voiceless"},
        "faa": {"place_simplified": "labial", "place": "labiodental", "manner": "fricative", "voicing": "voiceless"},
        "baa": {"place_simplified": "labial", "place": "bilabial", "manner": "stop", "voicing": "voiced"},
        "naa": {"place_simplified": "coronal", "place": "alveolar", "manner": "nasal", "voicing": "voiced"},
        "raa": {"place_simplified": "coronal", "place": "alveolar", "manner": "approximant", "voicing": "voiced"},
        "laa": {"place_simplified": "coronal", "place": "alveolar", "manner": "lateral", "voicing": "voiced"},
        "yaa": {"place_simplified": "coronal", "place": "palatal", "manner": "approximant", "voicing": "voiced"},
        "zaa": {"place_simplified": "coronal", "place": "alveolar", "manner": "fricative", "voicing": "voiced"},
    }

    return pd.DataFrame(cv_reference).T

def get_phoneme_reference():
    vowel_reference = {
        'AA': {'height': 'low', 'backness': 'back', 'roundedness': 'unrounded', 'tenseness': 'tense', 'diphthongization': 'monophthong'},
        'AE': {'height': 'low', 'backness': 'front', 'roundedness': 'unrounded', 'tenseness': 'lax', 'diphthongization': 'monophthong'},
        'AH': {'height': 'mid', 'backness': 'central', 'roundedness': 'unrounded', 'tenseness': 'lax', 'diphthongization': 'monophthong'},
        'AO': {'height': 'low-mid', 'backness': 'back', 'roundedness': 'rounded', 'tenseness': 'tense', 'diphthongization': 'monophthong'},
        'AW': {'height': 'low', 'backness': 'back', 'roundedness': 'rounded', 'tenseness': 'tense', 'diphthongization': 'diphthong'},
        'AY': {'height': 'low', 'backness': 'central', 'roundedness': 'unrounded', 'tenseness': 'tense', 'diphthongization': 'diphthong'},
        'EH': {'height': 'mid', 'backness': 'front', 'roundedness': 'unrounded', 'tenseness': 'lax', 'diphthongization': 'monophthong'},
        'ER': {'height': 'mid', 'backness': 'central', 'roundedness': 'unrounded', 'tenseness': 'tense', 'diphthongization': 'monophthong'},
        'EY': {'height': 'mid', 'backness': 'front', 'roundedness': 'unrounded', 'tenseness': 'tense', 'diphthongization': 'diphthong'},
        'IH': {'height': 'high', 'backness': 'front', 'roundedness': 'unrounded', 'tenseness': 'lax', 'diphthongization': 'monophthong'},
        'IY': {'height': 'high', 'backness': 'front', 'roundedness': 'unrounded', 'tenseness': 'tense', 'diphthongization': 'monophthong'},
        'OW': {'height': 'mid', 'backness': 'back', 'roundedness': 'rounded', 'tenseness': 'tense', 'diphthongization': 'diphthong'},
        'OY': {'height': 'mid', 'backness': 'back', 'roundedness': 'rounded', 'tenseness': 'tense', 'diphthongization': 'diphthong'},
        'UH': {'height': 'high-mid', 'backness': 'back', 'roundedness': 'rounded', 'tenseness': 'lax', 'diphthongization': 'monophthong'},
        'UW': {'height': 'high', 'backness': 'back', 'roundedness': 'rounded', 'tenseness': 'tense', 'diphthongization': 'monophthong'}
    }

    consonant_reference = {
        "P": {"place_simplified": "labial", "place": "bilabial", "manner": "stop", "voicing": "voiceless"},
        "B": {"place_simplified": "labial", "place": "bilabial", "manner": "stop", "voicing": "voiced"},
        "CH": {"place_simplified": "coronal", "place": "postalveolar", "manner": "affricate", "voicing": "voiceless"},
        "D": {"place_simplified": "coronal", "place": "alveolar", "manner": "stop", "voicing": "voiced"},
        "DH": {"place_simplified": "dental", "place": "dental", "manner": "fricative", "voicing": "voiced"},
        "F": {"place_simplified": "labial", "place": "labiodental", "manner": "fricative", "voicing": "voiceless"},
        "G": {"place_simplified": "velar", "place": "velar", "manner": "stop", "voicing": "voiced"},
        "HH": {"place_simplified": "laryngeal", "place": "glottal", "manner": "fricative", "voicing": "voiceless"},
        "JH": {"place_simplified": "coronal", "place": "postalveolar", "manner": "affricate", "voicing": "voiced"},
        "K": {"place_simplified": "velar", "place": "velar", "manner": "stop", "voicing": "voiceless"},
        "L": {"place_simplified": "coronal", "place": "alveolar", "manner": "lateral", "voicing": "voiced"},
        "M": {"place_simplified": "labial", "place": "bilabial", "manner": "nasal", "voicing": "voiced"},
        "N": {"place_simplified": "coronal", "place": "alveolar", "manner": "nasal", "voicing": "voiced"},
        "NG": {"place_simplified": "velar", "place": "velar", "manner": "nasal", "voicing": "voiced"},
        "R": {"place_simplified": "coronal", "place": "alveolar", "manner": "approximant", "voicing": "voiced"},
        "S": {"place_simplified": "coronal", "place": "alveolar", "manner": "fricative", "voicing": "voiceless"},
        "SH": {"place_simplified": "coronal", "place": "postalveolar", "manner": "fricative", "voicing": "voiceless"},
        "T": {"place_simplified": "coronal", "place": "alveolar", "manner": "stop", "voicing": "voiceless"},
        "TH": {"place_simplified": "dental", "place": "dental", "manner": "fricative", "voicing": "voiceless"},
        "V": {"place_simplified": "labial", "place": "labiodental", "manner": "fricative", "voicing": "voiced"},
        "W": {"place_simplified": "labial", "place": "bilabial", "manner": "approximant", "voicing": "voiced"},
        "Y": {"place_simplified": "coronal", "place": "palatal", "manner": "approximant", "voicing": "voiced"},
        "Z": {"place_simplified": "coronal", "place": "alveolar", "manner": "fricative", "voicing": "voiced"},
        "ZH": {"place_simplified": "coronal", "place": "postalveolar", "manner": "fricative", "voicing": "voiced"}
    }

    return pd.DataFrame(vowel_reference).T, pd.DataFrame(consonant_reference).T