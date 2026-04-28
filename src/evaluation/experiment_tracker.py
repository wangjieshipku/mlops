"""
Experiment Tracking Module
==========================
Local experiment tracking with MLflow integration.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient


class ExperimentTracker:
    """
    Experiment tracking using MLflow for local tracking.
    """

    def __init__(
        self,
        tracking_uri: str = "experiments/mlruns",
        experiment_name: str = "iris_classification",
        log_models: bool = True,
        log_artifacts: bool = True,
    ):
        """
        Initialize ExperimentTracker.

        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Name of the experiment
            log_models: Whether to log models
            log_artifacts: Whether to log artifacts
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.log_models = log_models
        self.log_artifacts = log_artifacts
        self.run_id: Optional[str] = None
        self.experiment_id: Optional[str] = None

        self._setup_mlflow()

    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        # Create directory if not exists
        Path(self.tracking_uri).mkdir(parents=True, exist_ok=True)

        # Set tracking URI
        mlflow.set_tracking_uri(f"file:///{os.path.abspath(self.tracking_uri)}")

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except mlflow.exceptions.MlflowException:
            self.experiment_id = mlflow.get_experiment_by_name(
                self.experiment_name
            ).experiment_id

        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow experiment '{self.experiment_name}' initialized")

    def start_run(self, run_name: Optional[str] = None, tags: Dict[str, str] = None):
        """
        Start a new MLflow run.

        Args:
            run_name: Name for this run
            tags: Tags to add to the run
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.active_run = mlflow.start_run(run_name=run_name)
        self.run_id = self.active_run.info.run_id

        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        logger.info(f"Started MLflow run: {run_name} (ID: {self.run_id})")

    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        mlflow.end_run(status=status)
        logger.info(f"Ended MLflow run with status: {status}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.

        Args:
            params: Dictionary of parameters
        """
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)

        for key, value in flat_params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Could not log param {key}: {e}")

        logger.info(f"Logged {len(flat_params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Step number (optional)
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    mlflow.log_metric(key, value, step=step)
                except Exception as e:
                    logger.warning(f"Could not log metric {key}: {e}")

        logger.info(f"Logged {len(metrics)} metrics")

    def log_artifact(self, artifact_path: str, artifact_dir: str = None):
        """
        Log an artifact.

        Args:
            artifact_path: Path to the artifact
            artifact_dir: Artifact subdirectory
        """
        if self.log_artifacts:
            mlflow.log_artifact(artifact_path, artifact_dir)
            logger.info(f"Logged artifact: {artifact_path}")

    def log_model(self, model, model_name: str, signature=None):
        """
        Log a model.

        Args:
            model: Model object
            model_name: Name for the model
            signature: MLflow model signature
        """
        if self.log_models:
            mlflow.sklearn.log_model(model, model_name, signature=signature)
            logger.info(f"Logged model: {model_name}")

    def log_figure(self, figure, artifact_file: str):
        """
        Log a matplotlib figure.

        Args:
            figure: Matplotlib figure
            artifact_file: Filename for the artifact
        """
        mlflow.log_figure(figure, artifact_file)
        logger.info(f"Logged figure: {artifact_file}")

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """
        Log a dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Filename for the artifact
        """
        mlflow.log_dict(dictionary, artifact_file)
        logger.info(f"Logged dict: {artifact_file}")

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def get_run_info(self) -> Dict[str, Any]:
        """
        Get information about current run.

        Returns:
            Dictionary with run information
        """
        if self.run_id is None:
            return {}

        client = MlflowClient()
        run = client.get_run(self.run_id)

        return {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "artifact_uri": run.info.artifact_uri,
        }

    def get_best_run(
        self, metric: str = "accuracy", ascending: bool = False
    ) -> Dict[str, Any]:
        """
        Get the best run based on a metric.

        Args:
            metric: Metric to compare
            ascending: Sort in ascending order

        Returns:
            Best run information
        """
        client = MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            return {}

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        if not runs:
            return {}

        best_run = runs[0]
        return {
            "run_id": best_run.info.run_id,
            "run_name": best_run.info.run_name,
            "metrics": best_run.data.metrics,
            "params": best_run.data.params,
            "artifact_uri": best_run.info.artifact_uri,
        }

    def list_experiments(self) -> List[Dict[str, str]]:
        """
        List all experiments.

        Returns:
            List of experiment information
        """
        client = MlflowClient()
        experiments = client.search_experiments()

        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
            }
            for exp in experiments
        ]
