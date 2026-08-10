from unittest import TestCase
from unittest.mock import MagicMock

from bxagent.evaluation import EvaluationResult
from bxagent.evaluation.pipefilter import EvaluationPipe, EvaluationFilter


class TestPipefilter(TestCase):
    def setUp(self):
        self.pipe = EvaluationPipe()
        self.fake_results = [
            EvaluationResult(
                content="result1", metadata={"success": True, "filter_prop_1": 1}
            ),
            EvaluationResult(
                content="result2", metadata={"success": False, "filter_prop_1": 2}
            ),
            EvaluationResult(
                content="result3", metadata={"success": False, "filter_prop_1": 1}
            ),
            EvaluationResult(
                content="result4", metadata={"success": True, "filter_prop_2": None}
            ),
            EvaluationResult(
                content="result5", metadata={"success": True, "filter_prop_2": "1"}
            ),
            EvaluationResult(
                content="result6", metadata={"success": True, "filter_prop_3": 1}
            ),
            EvaluationResult(
                content="result7", metadata={"success": True, "filter_prop_3": 1}
            ),
        ]

    def test_filter_results__execute_all_filters(self):
        filter1 = MagicMock(spec=EvaluationFilter, return_value=[
            self.fake_results[0], self.fake_results[1], self.fake_results[2]
        ])
        filter2 = MagicMock(spec=EvaluationFilter, return_value=[
            self.fake_results[0], self.fake_results[1]
        ])
        pipe = EvaluationPipe() | filter1 | filter2

        results = pipe.filter_results(self.fake_results)

        self.assertListEqual(results, [self.fake_results[0], self.fake_results[1]])
