from unittest import TestCase
from unittest.mock import MagicMock

from bxagent.evaluation import ValidationResult
from bxagent.evaluation.pipefilter import ValidationPipe, ValidationFilter


class TestPipefilter(TestCase):
    def setUp(self):
        self.pipe = ValidationPipe()
        self.fake_results = [
            ValidationResult(
                content="result1", metadata={"success": True, "filter_prop_1": 1}
            ),
            ValidationResult(
                content="result2", metadata={"success": False, "filter_prop_1": 2}
            ),
            ValidationResult(
                content="result3", metadata={"success": False, "filter_prop_1": 1}
            ),
            ValidationResult(
                content="result4", metadata={"success": True, "filter_prop_2": None}
            ),
            ValidationResult(
                content="result5", metadata={"success": True, "filter_prop_2": "1"}
            ),
            ValidationResult(
                content="result6", metadata={"success": True, "filter_prop_3": 1}
            ),
            ValidationResult(
                content="result7", metadata={"success": True, "filter_prop_3": 1}
            ),
        ]

    def test_filter_results__execute_all_filters(self):
        filter1 = MagicMock(spec=ValidationFilter, return_value=[
            self.fake_results[0], self.fake_results[1], self.fake_results[2]
        ])
        filter2 = MagicMock(spec=ValidationFilter, return_value=[
            self.fake_results[0], self.fake_results[1]
        ])
        pipe = ValidationPipe() | filter1 | filter2

        results = pipe.filter_results(self.fake_results)

        self.assertListEqual(results, [self.fake_results[0], self.fake_results[1]])
