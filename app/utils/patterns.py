"""
Regex patterns and label mappings for entity extraction.
"""

import re

# ============================================================================
# Regex patterns for IC, phone, email
# ============================================================================
REGEX_PATTERNS = {
    # Malaysian IC: YYMMDD-SS-NNNN or 12 digits
    'ic': re.compile(r'\b\d{6}-\d{2}-\d{4}\b|\b\d{12}\b'),
    
    # Malaysian phone: +60/60/0 followed by 1X-XXX-XXXX or landline 0X-XXXXXXX
    'phone': re.compile(r'\b(?:\+?6?0)?1[0-9]-?\d{3,4}-?\d{4}\b|\b0[3-9]-?\d{7,8}\b'),
    
    # Email: standard email pattern
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
}


# ============================================================================
# Label mappings
# ============================================================================
# Model outputs LABEL_0, LABEL_1, LABEL_2
# Map to semantic entity types
LABEL_TO_TYPE = {
    'LABEL_1': 'name',
    'LABEL_2': 'address',
}

# Human-readable label names
LABEL_TO_READABLE = {
    'LABEL_0': 'O',        # Outside / Non-entity
    'LABEL_1': 'name',     # Person name
    'LABEL_2': 'address',  # Address
}


def extract_regex_entities(text: str) -> dict:
    """
    Extract entities using regex patterns.
    
    Args:
        text: Input text to search
        
    Returns:
        Dictionary with keys 'ic', 'phone', 'email' containing lists of matches
    """
    entities = {'ic': [], 'phone': [], 'email': []}
    
    for entity_type, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            entities[entity_type].append(match.group())
    
    return entities
