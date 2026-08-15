"""Mode 1: prediction-error anomaly vs EWMA/CUSUM on TOW-P-shaped residuals."""

from sandbox.mode1_anomaly.champion import ChampionDetector, TimeOfWeekTempBaseline
from sandbox.mode1_anomaly.challenger import NextStepSurpriseDetector
from sandbox.mode1_anomaly.evaluate import compare_detectors, detector_metrics
from sandbox.mode1_anomaly.generate import STEPS_PER_DAY, PlantConfig, generate_plant

__all__ = [
    "STEPS_PER_DAY",
    "PlantConfig",
    "generate_plant",
    "TimeOfWeekTempBaseline",
    "ChampionDetector",
    "NextStepSurpriseDetector",
    "detector_metrics",
    "compare_detectors",
]
