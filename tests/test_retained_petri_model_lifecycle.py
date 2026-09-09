"""Run with the pinned Petri runtime; these tests need no network or weights."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('inspect_ai'), 'requires the isolated Petri runtime')
class ModelLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_second_organism_does_not_reuse_first_target(self):
        from inspect_ai.model import ChatMessageUser
        from slc.retained_petri_target import make_target_model

        class Engine:
            def __init__(self, text):
                self.text = text

            def generate(self, *args):
                return {'text': self.text, 'input_tokens': 1, 'output_tokens': 1,
                        'stop_reason': 'stop'}

        first = make_target_model(Engine('first organism'))
        first_output = await first.generate([ChatMessageUser(content='Hello')])
        second = make_target_model(Engine('second organism'))
        second_output = await second.generate([ChatMessageUser(content='Hello')])
        self.assertIsNot(first, second)
        self.assertEqual(first_output.completion, 'first organism')
        self.assertEqual(second_output.completion, 'second organism')
