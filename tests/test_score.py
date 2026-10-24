import unittest

from sge_pipline.evaluate import answers_match, normalize_math


class ScoreTest(unittest.TestCase):
    def test_boxed(self):
        self.assertTrue(answers_match(r"reasoning \boxed{721}", "721"))

    def test_fraction(self):
        self.assertTrue(answers_match("1/2", "0.5"))

    def test_zero_pad(self):
        self.assertTrue(answers_match("025", "25"))

    def test_mismatch(self):
        self.assertFalse(answers_match("3", "4"))

    def test_answer_prefix(self):
        self.assertEqual(normalize_math("The answer is 025."), "025")

    def test_none(self):
        self.assertEqual(normalize_math(None), "")


class RouteTest(unittest.TestCase):
    def test_vector_shape(self):
        try:
            from sge_pipline.sge import route_detail
            decision, probability, features = route_detail("What is 2 + 2?")
        except (ImportError, ModuleNotFoundError, OSError):
            self.skipTest("router dependencies are not installed")
        self.assertEqual(len(features), 10)
        self.assertIn(decision, ("LLM", "LRM"))
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
