import importlib_resources

from xlmexlab.evaluator import Evaluator


def test_evaluator():
    dataset_path: str = str(
        importlib_resources.files("xlmexlab") / "resources" / "evaluation_test.txt"
    )
    evaluator: Evaluator = Evaluator(reference_dataset_path=dataset_path)
    actions_eval = evaluator.evaluate_actions(dataset_path)
    sequence_eval = evaluator.evaluate_actions_order(dataset_path)
    assert actions_eval["precision"] == 1.0
    assert actions_eval["recall"] == 1.0
    assert actions_eval["f-score"] == 1.0
    assert sequence_eval == {"%%extra": 0.0, "%missing": 0.0, "accuracy": 1.0}
