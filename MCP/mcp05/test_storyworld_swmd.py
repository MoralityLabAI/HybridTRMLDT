import unittest

from .storyworld_swmd import evaluate, parse_scene, profile_pair, rehearse, render_card, world_ids


class CompactCards(unittest.TestCase):
    def test_profile_guard_preserves_state_dependence(self):
        data = [[[0, 25, 0, 0], [5, -4, 0]], [[0, -25, 0, 0], [-5, 4, 0]]]
        self.assertEqual(profile_pair(data), ((-5, 4, 0), (5, -4, 0)))
        data[1][0] = [60, 25, 0, 0]
        with self.assertRaisesRegex(ValueError, "each reaction must be favored"):
            profile_pair(data)

    def test_round_trip_preserves_the_action_math(self):
        card = {"cast": [{"id": "char_probe"}], "endings": [{"id": f"end_{i}"} for i in range(4)]}
        data = {"title": "Round trip", "text": "This is a test fixture with a visible narration line.",
                "art": "A test grid", "options": []}
        for weight, stat in ((8, 1), (11, 2), (14, 3)):
            positive = [0, 0, 0, 0]
            positive[stat] = weight
            negative = [-x for x in positive]
            data["options"].append([f"Option {stat}", ["enc_02", "Positive probe", positive, [weight, -4, 0]],
                                    ["enc_03", "Negative probe", negative, [-weight, 4, 0]]])
        compiled = render_card(data, card, "enc_01")
        scene = parse_scene(compiled, card, "enc_01")
        self.assertEqual(len(world_ids(card)), 20)
        self.assertEqual(scene["text"], data["text"])
        reaction = scene["options"][0]["reactions"][0]
        self.assertEqual(reaction["effects"], {"trust": 0.08, "nerve": -0.04})
        self.assertAlmostEqual(evaluate(reaction["score"], {"trust": 0.5, "nerve": 0, "care": 0}), .04)
        with self.assertRaisesRegex(ValueError, "invalid forward target"):
            parse_scene(compiled.replace("-> enc_02", "-> enc_01", 1), card, "enc_01")

    def test_rehearsal_replays_clamped_effects_and_records_a_witness(self):
        yes = {"id": "positive", "to": "end", "score": [0, .2, 0, 0], "effects": {"trust": .2}}
        no = {"id": "negative", "to": "end", "score": [0, -.2, 0, 0], "effects": {"trust": -.2}}
        start = {"id": "start", "options": [{"id": "choose", "reactions": [yes, no]}]}
        result = rehearse([start, {"id": "end", "options": []}], {"trust": .95, "nerve": 0, "care": 0}, runs=4)
        self.assertEqual(result["endings"], {"end": 4})
        self.assertEqual(result["clamped_effects"], 4)
        witness = result["scene_witnesses"]["end"][0]
        self.assertEqual((witness["reaction"], witness["after"]["trust"]), ("positive", 1.0))
