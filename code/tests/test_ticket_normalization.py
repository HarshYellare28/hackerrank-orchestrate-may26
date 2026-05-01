import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from ticket_normalization import (
    is_acknowledgement,
    normalize_company_hint,
    normalize_csv_row,
    normalize_rows,
    normalize_text,
)


class TicketNormalizationTest(unittest.TestCase):
    def test_normalizes_whitespace_and_combines_subject_before_issue(self):
        ticket = normalize_csv_row(
            {
                "Issue": "  I cannot\naccess\tClaude   workspace. ",
                "Subject": " Access lost ",
                "Company": " Claude ",
            },
            row_index=7,
        )

        self.assertEqual(ticket.row_index, 7)
        self.assertEqual(ticket.issue, "I cannot access Claude workspace.")
        self.assertEqual(ticket.subject, "Access lost")
        self.assertEqual(ticket.company_raw, "Claude")
        self.assertEqual(ticket.company_hint, "claude")
        self.assertEqual(ticket.text, "Access lost I cannot access Claude workspace.")
        self.assertEqual(ticket.searchable_text, "access lost i cannot access claude workspace.")
        self.assertEqual(ticket.tokens, ("access", "lost", "i", "cannot", "access", "claude", "workspace"))

    def test_handles_company_none_as_missing_hint(self):
        ticket = normalize_csv_row(
            {"Issue": "site is down", "Subject": "", "Company": " None "},
            row_index=1,
        )

        self.assertIsNone(ticket.company_hint)
        self.assertFalse(ticket.is_empty)

    def test_supports_case_insensitive_headers(self):
        ticket = normalize_csv_row(
            {"issue": "How do I dispute a charge?", "subject": "Help", "company": "VISA"},
            row_index=2,
        )

        self.assertEqual(ticket.company_hint, "visa")
        self.assertEqual(ticket.text, "Help How do I dispute a charge?")

    def test_detects_empty_and_acknowledgement_rows(self):
        empty_ticket = normalize_csv_row({"Issue": " ", "Subject": "", "Company": ""}, row_index=1)
        thanks_ticket = normalize_csv_row(
            {"Issue": "Thank you for helping me", "Subject": "", "Company": "None"},
            row_index=2,
        )

        self.assertTrue(empty_ticket.is_empty)
        self.assertFalse(empty_ticket.is_acknowledgement)
        self.assertTrue(thanks_ticket.is_acknowledgement)

    def test_batch_normalization_uses_one_based_indexes(self):
        tickets = normalize_rows(
            [
                {"Issue": "first", "Subject": "", "Company": ""},
                {"Issue": "second", "Subject": "", "Company": ""},
            ]
        )

        self.assertEqual([ticket.row_index for ticket in tickets], [1, 2])

    def test_helper_functions_are_stable(self):
        self.assertEqual(normalize_text(" a\n\nb\tc "), "a b c")
        self.assertEqual(normalize_company_hint("Anthropic Claude"), "claude")
        self.assertTrue(is_acknowledgement(" Thanks! "))


if __name__ == "__main__":
    unittest.main()
