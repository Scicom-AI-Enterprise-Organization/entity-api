"""
Test app/utils modules: text.py and patterns.py
"""
import pytest


class TestTextModule:
    """Test app/utils/text.py - BPE token merging"""
    
    def test_merge_bpe_tokens_basic(self):
        """Test basic BPE merging with GPT-style prefix."""
        from app.utils.text import merge_bpe_tokens_tagging
        
        tokens = ["Hello", "Ġworld", "Ġfrom", "ĠMalaysia"]
        labels = ["LABEL_0", "LABEL_0", "LABEL_1", "LABEL_2"]
        
        words, word_labels = merge_bpe_tokens_tagging(tokens, labels, prefix_char='Ġ')
        
        assert words == ["Hello", "world", "from", "Malaysia"]
        assert word_labels == ["LABEL_0", "LABEL_0", "LABEL_1", "LABEL_2"]
    
    def test_merge_sentencepiece_tokens(self):
        """Test merging SentencePiece tokens (▁ prefix)."""
        from app.utils.text import merge_bpe_tokens_tagging
        
        tokens = ["▁Hello", "▁world"]
        labels = ["LABEL_0", "LABEL_1"]
        
        words, word_labels = merge_bpe_tokens_tagging(tokens, labels)
        
        assert words == ["Hello", "world"]
        assert word_labels == ["LABEL_0", "LABEL_1"]
    
    def test_merge_subword_continuation(self):
        """Test merging subword tokens (continuation without prefix)."""
        from app.utils.text import merge_bpe_tokens_tagging
        
        tokens = ["ĠKuala", "Lumpur"]  # Lumpur is continuation
        labels = ["LABEL_2", "LABEL_2"]
        
        words, word_labels = merge_bpe_tokens_tagging(tokens, labels, prefix_char='Ġ')
        
        assert len(words) == 1
        assert words[0] == "KualaLumpur"
        assert word_labels[0] == "LABEL_2"
    
    def test_skip_special_tokens(self):
        """Test that special tokens are filtered out."""
        from app.utils.text import merge_bpe_tokens_tagging
        
        tokens = ["<s>", "Hello", "Ġworld", "</s>"]
        labels = ["O", "O", "O", "O"]
        
        words, word_labels = merge_bpe_tokens_tagging(tokens, labels, prefix_char='Ġ')
        
        assert "<s>" not in words
        assert "</s>" not in words
        assert "Hello" in words
        assert "world" in words
    
    def test_preserve_first_subword_label(self):
        """Test that merged word keeps first subword's label."""
        from app.utils.text import merge_bpe_tokens_tagging
        
        tokens = ["ĠKuala", "Lumpur", "Ġis"]
        labels = ["LABEL_2", "LABEL_1", "LABEL_0"]  # Different labels for subwords
        
        words, word_labels = merge_bpe_tokens_tagging(tokens, labels, prefix_char='Ġ')
        
        # KualaLumpur should keep LABEL_2 (first subword)
        assert word_labels[0] == "LABEL_2"
    
    def test_empty_input(self):
        """Test with empty input."""
        from app.utils.text import merge_bpe_tokens_tagging
        
        words, labels = merge_bpe_tokens_tagging([], [])
        
        assert words == []
        assert labels == []
    
    def test_custom_rejected_tokens(self):
        """Test with custom rejected token list."""
        from app.utils.text import merge_bpe_tokens_tagging
        
        tokens = ["Hello", "CUSTOM_TOKEN", "world"]
        labels = ["O", "O", "O"]
        
        words, word_labels = merge_bpe_tokens_tagging(
            tokens, labels, rejected=["CUSTOM_TOKEN"]
        )
        
        assert "CUSTOM_TOKEN" not in words
    
    def test_multiple_consecutive_subwords(self):
        """Test multiple consecutive subwords merge correctly."""
        from app.utils.text import merge_bpe_tokens_tagging
        
        tokens = ["ĠMuham", "mad", "Ah", "mad"]
        labels = ["LABEL_1", "LABEL_1", "LABEL_1", "LABEL_1"]
        
        words, word_labels = merge_bpe_tokens_tagging(tokens, labels, prefix_char='Ġ')
        
        # All should merge into one word
        assert len(words) == 1
        assert words[0] == "MuhammadAhmad"


class TestPatternsModule:
    """Test app/utils/patterns.py - Regex patterns and label mappings"""
    
    def test_regex_patterns_exist(self):
        """Test that all required regex patterns are defined."""
        from app.utils.patterns import REGEX_PATTERNS
        
        assert 'ic' in REGEX_PATTERNS
        assert 'phone' in REGEX_PATTERNS
        assert 'email' in REGEX_PATTERNS
    
    def test_regex_patterns_are_compiled(self):
        """Test that patterns are compiled regex objects."""
        from app.utils.patterns import REGEX_PATTERNS
        import re
        
        for name, pattern in REGEX_PATTERNS.items():
            assert isinstance(pattern, re.Pattern), f"{name} should be compiled regex"
    
    def test_label_to_type_mapping(self):
        """Test LABEL_TO_TYPE dictionary."""
        from app.utils.patterns import LABEL_TO_TYPE
        
        assert LABEL_TO_TYPE['LABEL_1'] == 'name'
        assert LABEL_TO_TYPE['LABEL_2'] == 'address'
    
    def test_label_to_readable_mapping(self):
        """Test LABEL_TO_READABLE dictionary."""
        from app.utils.patterns import LABEL_TO_READABLE
        
        assert LABEL_TO_READABLE['LABEL_0'] == 'O'
        assert LABEL_TO_READABLE['LABEL_1'] == 'name'
        assert LABEL_TO_READABLE['LABEL_2'] == 'address'
    
    def test_extract_ic_with_hyphens(self):
        """Test extracting IC number with hyphens."""
        from app.utils.patterns import extract_regex_entities
        
        text = "IC saya 900101-14-5678"
        entities = extract_regex_entities(text)
        
        assert len(entities['ic']) == 1
        assert entities['ic'][0] == "900101-14-5678"
    
    def test_extract_ic_without_hyphens(self):
        """Test extracting IC number without hyphens (12 digits)."""
        from app.utils.patterns import extract_regex_entities
        
        text = "IC saya 900101145678"
        entities = extract_regex_entities(text)
        
        assert len(entities['ic']) == 1
        assert entities['ic'][0] == "900101145678"
    
    def test_extract_mobile_phone(self):
        """Test extracting mobile phone numbers."""
        from app.utils.patterns import extract_regex_entities
        
        text = "hubungi 0123456789"
        entities = extract_regex_entities(text)
        
        assert len(entities['phone']) >= 1
    
    def test_extract_phone_with_hyphens(self):
        """Test extracting phone with hyphens."""
        from app.utils.patterns import extract_regex_entities
        
        text = "call 012-345-6789"
        entities = extract_regex_entities(text)
        
        assert len(entities['phone']) >= 1
    
    def test_extract_landline(self):
        """Test extracting landline phone numbers."""
        from app.utils.patterns import extract_regex_entities
        
        text = "telefon pejabat 03-12345678"
        entities = extract_regex_entities(text)
        
        assert len(entities['phone']) >= 1
    
    def test_extract_email(self):
        """Test extracting email addresses."""
        from app.utils.patterns import extract_regex_entities
        
        text = "email saya test@gmail.com"
        entities = extract_regex_entities(text)
        
        assert len(entities['email']) == 1
        assert entities['email'][0] == "test@gmail.com"
    
    def test_extract_multiple_emails(self):
        """Test extracting multiple email addresses."""
        from app.utils.patterns import extract_regex_entities
        
        text = "email test@gmail.com atau ali@example.com"
        entities = extract_regex_entities(text)
        
        assert len(entities['email']) == 2
        assert "test@gmail.com" in entities['email']
        assert "ali@example.com" in entities['email']
    
    def test_extract_multiple_entity_types(self):
        """Test extracting all entity types from one text."""
        from app.utils.patterns import extract_regex_entities
        
        text = "IC 900101-14-5678, hubungi 0123456789, email ali@test.com"
        entities = extract_regex_entities(text)
        
        assert len(entities['ic']) == 1
        assert len(entities['phone']) == 1
        assert len(entities['email']) == 1
    
    def test_no_entities_found(self):
        """Test text with no regex entities."""
        from app.utils.patterns import extract_regex_entities
        
        text = "Nama saya Ahmad dari Kuala Lumpur"
        entities = extract_regex_entities(text)
        
        assert len(entities['ic']) == 0
        assert len(entities['phone']) == 0
        assert len(entities['email']) == 0
