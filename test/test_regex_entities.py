"""
Test regex entity extraction (IC, phone, email).
"""
import unittest
from flash_infer_encoder_non_causal.main import REGEX_PATTERNS, extract_regex_entities


class TestRegexEntities(unittest.TestCase):
    def test_ic_extraction(self):
        """Test Malaysian IC number extraction."""
        text = "IC saya 900101-14-5678"
        entities = extract_regex_entities(text)
        
        ic_entities = [e for e in entities if e['label'] == 'IC']
        self.assertEqual(len(ic_entities), 1, "Should extract one IC")
        self.assertEqual(ic_entities[0]['text'], "900101-14-5678", "IC format mismatch")

    def test_ic_without_hyphens(self):
        """Test IC without hyphens."""
        text = "IC saya 900101145678"
        entities = extract_regex_entities(text)
        
        ic_entities = [e for e in entities if e['label'] == 'IC']
        self.assertEqual(len(ic_entities), 1, "Should extract IC without hyphens")
        self.assertEqual(ic_entities[0]['text'], "900101145678")

    def test_phone_extraction(self):
        """Test phone number extraction."""
        texts = [
            "hubungi 0123456789",
            "call 016-2587806",
            "telefon +60123456789",
        ]
        
        for text in texts:
            entities = extract_regex_entities(text)
            phone_entities = [e for e in entities if e['label'] == 'PHONE']
            self.assertGreater(len(phone_entities), 0, f"No phone found in: {text}")

    def test_email_extraction(self):
        """Test email extraction."""
        text = "email saya test@gmail.com atau ali@example.com"
        entities = extract_regex_entities(text)
        
        email_entities = [e for e in entities if e['label'] == 'EMAIL']
        self.assertEqual(len(email_entities), 2, "Should extract two emails")
        self.assertIn("test@gmail.com", [e['text'] for e in email_entities])
        self.assertIn("ali@example.com", [e['text'] for e in email_entities])

    def test_multiple_entity_types(self):
        """Test extracting multiple entity types from one text."""
        text = "Nama saya Ali, IC 900101-14-5678, hubungi 0123456789 atau email ali@test.com"
        entities = extract_regex_entities(text)
        
        ic_count = len([e for e in entities if e['label'] == 'IC'])
        phone_count = len([e for e in entities if e['label'] == 'PHONE'])
        email_count = len([e for e in entities if e['label'] == 'EMAIL'])
        
        self.assertEqual(ic_count, 1, "Should extract one IC")
        self.assertEqual(phone_count, 1, "Should extract one phone")
        self.assertEqual(email_count, 1, "Should extract one email")

    def test_entity_positions(self):
        """Test that entity positions (start, end) are correct."""
        text = "IC saya 900101-14-5678"
        entities = extract_regex_entities(text)
        
        ic_entity = [e for e in entities if e['label'] == 'IC'][0]
        extracted_text = text[ic_entity['start']:ic_entity['end']]
        
        self.assertEqual(extracted_text, "900101-14-5678", 
                        "Extracted text should match position")


if __name__ == "__main__":
    unittest.main()
