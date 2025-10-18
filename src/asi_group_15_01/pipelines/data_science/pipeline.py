"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline
from .nodes import load_raw, basic_clean, train_test_split, train_baseline, evaluate

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=load_raw,
                inputs="params:data.raw_csv",
                outputs="raw_data",
                name="load",
            ),
            Node(
                func=basic_clean,
                inputs="raw_data",
                outputs="cleaned_data",
                name="clean",
            ),
            Node(
                func=train_test_split,
                inputs=["cleaned_data", "params:split"],
                outputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                ],
                name="split",
            ),
            Node(
                func=train_baseline,
                inputs=["X_train", "y_train", "params:model"],
                outputs="baseline_model",
                name="train_baseline",
            ),
            Node(
                func=evaluate,
                inputs=["baseline_model", "X_test", "y_test"],
                outputs="evaluation_metrics",
                name="evaluate",
            ),
        ]
    )