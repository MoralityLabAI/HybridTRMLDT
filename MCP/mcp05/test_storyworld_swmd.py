import unittest

from .inky_storyworld import apply_player_fixes, check_art_map, check_outline, check_profiles, duplicated, encounter_card
from .storyworld_swmd import evaluate, parse_scene, profile_pair, rehearse, render_card, world_ids


class CompactCards(unittest.TestCase):
    def test_art_compositions_accept_exact_scene_ids_and_check_variation(self):
        card = {"cast": [{"id": "char_test"}], "endings": [{"id": f"end_{i}"} for i in range(4)]}
        scene_ids = world_ids(card)
        rows = {id_: ["#010203", "#040506", "#070809", i, 80, 60, "fixture", ["char_test"]]
                for i, id_ in enumerate(scene_ids, 1)}
        self.assertEqual(check_art_map(rows, card), rows)
        rows[scene_ids[-1]] = rows[scene_ids[0]]
        with self.assertRaisesRegex(ValueError, "distinct"):
            check_art_map(rows, card)

    def test_player_replacements_require_exact_counts(self):
        original = "a\r\na\r\nb\r\n"
        self.assertEqual(apply_player_fixes(original, {"replacements": [
            {"old": "a", "new": "c", "count": 2}]}), "c\r\nc\r\nb\r\n")
        with self.assertRaisesRegex(ValueError, "expected 1 exact occurrences"):
            apply_player_fixes(original, {"replacements": [{"old": "a", "new": "c"}]})

    def test_outline_requires_distinct_beats_and_scene_cues(self):
        card = {"endings": [{"id": f"end_{i}"} for i in range(4)]}
        rows = [[id_, f"Title {i}", f"Beat {i}"] for i, id_ in enumerate(world_ids(card))]
        self.assertEqual(check_outline(rows, card)["enc_01"], ("Title 0", "Beat 0"))
        rows[1][2] = rows[0][2]
        with self.assertRaisesRegex(ValueError, "distinct"):
            check_outline(rows, card)
        self.assertEqual(duplicated({"text": "new", "art": "Blue    grid"},
                                   [{"id": "enc_01", "text": "old", "art": "blue grid"}]), "enc_01")

    def test_compact_encounter_expands_text_and_targets_verbatim(self):
        source = ["title", "scene", "art", [["choice", "test profile", "enc_02", "reply A", "enc_03", "reply B"]]]
        card = encounter_card(source)
        self.assertEqual(card["text"], "scene")
        self.assertEqual(card["options"][0]["model"], "test profile")
        self.assertEqual(card["options"][0]["reactions"], [
            {"next": "enc_02", "reply": "reply A"}, {"next": "enc_03", "reply": "reply B"}])
        self.assertEqual(card, encounter_card({"title": "title", "sceneText": "scene", "artCue": "art",
                                              "options": source[3]}))

    def test_profile_guard_preserves_state_dependence(self):
        data = [[[0, 25, 0, 0], [5, -4, 0]], [[0, -25, 0, 0], [-5, 4, 0]]]
        self.assertEqual(profile_pair(data), ((-5, 4, 0), (5, -4, 0)))
        data[1][0] = [60, 25, 0, 0]
        with self.assertRaisesRegex(ValueError, "each reaction must be favored"):
            profile_pair(data)

    def test_compact_profile_mirrors_authored_weights_without_altering_effects(self):
        compact = {f"case_{i}": [[i, 0, -i], [i, -i, 0], [-i, i, 0]] for i in range(1, 7)}
        profiles = check_profiles({"profiles": compact})
        self.assertEqual(profiles["case_4"], [[[0, 4, 0, -4], [4, -4, 0]], [[0, -4, 0, 4], [-4, 4, 0]]])

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
        authored = {**card, "outline": {"enc_01": ("Outline title", "An authored beat")}}
        self.assertIn("# TITLE: Outline title\n", render_card(data, authored, "enc_01"))
        with self.assertRaisesRegex(ValueError, "invalid route"):
            parse_scene(compiled.replace("-> enc_02", "-> enc_99", 1), card, "enc_01")

    def test_rehearsal_replays_clamped_effects_and_records_a_witness(self):
        yes = {"id": "positive", "to": "end", "score": [0, .2, 0, 0], "effects": {"trust": .2}}
        no = {"id": "negative", "to": "end", "score": [0, -.2, 0, 0], "effects": {"trust": -.2}}
        start = {"id": "start", "options": [{"id": "choose", "reactions": [yes, no]}]}
        result = rehearse([start, {"id": "end", "options": []}], {"trust": .95, "nerve": 0, "care": 0}, runs=4)
        self.assertEqual(result["endings"], {"end": 4})
        self.assertEqual(result["clamped_effects"], 4)
        witness = result["scene_witnesses"]["end"][0]
        self.assertEqual((witness["reaction"], witness["after"]["trust"]), ("positive", 1.0))
