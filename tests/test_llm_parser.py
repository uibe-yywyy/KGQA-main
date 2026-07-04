import json
import unittest

from egc_tecqa.parser.llm_parser import _extract_json


class LLMParserTest(unittest.TestCase):
    def test_extract_json_plain(self):
        data = _extract_json('{"entities":["Iraq"],"relation_phrase":"visit"}')
        self.assertEqual(data["entities"], ["Iraq"])

    def test_extract_json_wrapped(self):
        data = _extract_json('Here is it:\n{"main_entity":"Iraq"}\nDone')
        self.assertEqual(data["main_entity"], "Iraq")


if __name__ == "__main__":
    unittest.main()

