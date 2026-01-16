"""
Test BPE token merging functionality.

Tests merging subword tokens back to words with labels,
similar to Malaya's merge_sentencepiece_tokens_tagging.
"""
import unittest
from flash_infer_encoder_non_causal.main import merge_bpe_tokens_tagging


class TestBPEMerging(unittest.TestCase):
    def test_merge_gpt_style_tokens(self):
        """Test merging GPT-style tokens (Ġ prefix)."""
        tokens = ["Hello", "Ġworld", "Ġfrom", "ĠMalaysia"]
        labels = ["LABEL_0", "LABEL_0", "LABEL_1", "LABEL_2"]
        
        words, word_labels = merge_bpe_tokens_tagging(tokens, labels, prefix_char='Ġ')
        
        expected_words = ["Hello", "world", "from", "Malaysia"]
        expected_labels = ["LABEL_0", "LABEL_0", "LABEL_1", "LABEL_2"]
        
        self.assertEqual(words, expected_words, "Words mismatch")
        self.assertEqual(word_labels, expected_labels, "Labels mismatch")

    def test_merge_sentencepiece_tokens(self):
        """Test merging SentencePiece tokens (▁ prefix)."""
        tokens = ["▁Hello", "▁world", "▁from", "▁Malaysia"]
        labels = ["LABEL_0", "LABEL_0", "LABEL_1", "LABEL_2"]
        
        words, word_labels = merge_bpe_tokens_tagging(tokens, labels, prefix_char='▁')
        
        expected_words = ["Hello", "world", "from", "Malaysia"]
        expected_labels = ["LABEL_0", "LABEL_0", "LABEL_1", "LABEL_2"]
        
        self.assertEqual(words, expected_words, "Words mismatch")
        self.assertEqual(word_labels, expected_labels, "Labels mismatch")

    def test_merge_subword_tokens(self):
        """Test merging subword tokens within same word."""
        tokens = ["ĠKuala", "Lumpur"]  # Lumpur is continuation
        labels = ["LABEL_2", "LABEL_2"]
        
        words, word_labels = merge_bpe_tokens_tagging(tokens, labels, prefix_char='Ġ')
        
        # Lumpur should be merged with Kuala
        self.assertEqual(len(words), 1, "Should merge into one word")
        self.assertEqual(words[0], "KualaLumpur", "Subword merge failed")
        self.assertEqual(word_labels[0], "LABEL_2", "Label should be preserved")

    def test_skip_special_tokens(self):
        """Test that special tokens are skipped."""
        tokens = ["<s>", "Hello", "Ġworld", "</s>"]
        labels = ["LABEL_0", "LABEL_0", "LABEL_0", "LABEL_0"]
        
        words, word_labels = merge_bpe_tokens_tagging(tokens, labels, prefix_char='Ġ')
        
        # Special tokens should be filtered out
        self.assertNotIn("<s>", words, "Special token <s> should be filtered")
        self.assertNotIn("</s>", words, "Special token </s> should be filtered")

    def test_preserve_first_subword_label(self):
        """Test that first subword's label is preserved for merged word."""
        tokens = ["ĠKuala", "Lumpur", "Ġis", "Ġnice"]
        labels = ["LABEL_2", "LABEL_1", "LABEL_0", "LABEL_0"]
        
        words, word_labels = merge_bpe_tokens_tagging(tokens, labels, prefix_char='Ġ')
        
        # Kuala Lumpur should take LABEL_2 (first subword's label)
        self.assertEqual(word_labels[0], "LABEL_2", 
                        "Merged word should take first subword's label")


if __name__ == "__main__":
    unittest.main()
