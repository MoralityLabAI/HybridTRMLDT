"""Logit acceptance comparator must fail closed on incomplete inputs."""
import unittest
from research.sovereign_learning.s0_trainability.parity_cli import compare


def rows():
    return {"vocab_size":3,"prompts":[
        {"id":f"p{i:03d}","input_token_ids":[110,i],
         "token_ids":[0,1,2],
         "logits":[float(i%3),float((i+1)%3),float((i+2)%3)]}
        for i in range(32)]}


class ParityTests(unittest.TestCase):
    def test_identical_full_rows_pass(self):
        receipt=compare(rows(),rows())
        self.assertEqual(receipt["status"],"PASS")
        self.assertEqual(receipt["top1_agreement"],1.0)
        self.assertEqual(receipt["worst_abs_logit_delta"],0)

    def test_top1_flip_fails(self):
        other=rows()
        other["prompts"][0]["logits"]=[100,-100,-100]
        self.assertEqual(compare(rows(),other)["status"],"FAIL")

    def test_logit_shift_fails_even_if_top1_matches(self):
        other=rows()
        other["prompts"][0]["logits"]=[v+1 for v in other["prompts"][0]["logits"]]
        self.assertEqual(compare(rows(),other)["status"],"FAIL")

    def test_input_tokens_must_match(self):
        other=rows()
        other["prompts"][0]["input_token_ids"]=[110,999]
        with self.assertRaises(ValueError):compare(rows(),other)

    def test_vocab_size_must_be_pinned(self):
        other=rows()
        other["vocab_size"]=4
        with self.assertRaises(ValueError):compare(rows(),other)
        with self.assertRaises(ValueError):compare(rows(),rows(),expected_vocab_size=248320)

    def test_prompt_and_vocab_order_must_match(self):
        other=rows()
        other["prompts"][0]["token_ids"]=[1,0,2]
        with self.assertRaises(ValueError):compare(rows(),other)
        with self.assertRaises(ValueError):compare(rows(),{"vocab_size":3,"prompts":other["prompts"][:31]})

    def test_invalid_or_truncated_logits_are_rejected(self):
        other=rows()
        other["prompts"][0]["logits"]=[0,1,float("nan")]
        with self.assertRaises(ValueError):compare(rows(),other)
        other=rows()
        other["prompts"][0]["logits"]=[1]
        with self.assertRaises(ValueError):compare(rows(),other)


if __name__ == "__main__":
    unittest.main()
