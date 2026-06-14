"""
Meta-Cognitive AGI Experiments
"""

from .meta_cognitive_agent import MetaCognitiveAgent, MetaCognitiveLayer, CognitiveState
from .recursive_improvement import RecursiveImprovementExperiment, run_recursive_experiment
from .meta_cognition_lenses import (
    MetaCognitionLens, InternalState, MetaCognitionLensEvaluator,
    StepwiseStateAggregator, IntrinsicRewardCalculator, MetaCognitionMetrics
)
from .game_theory_self_awareness import (
    GuessTwoThirdsGame, RationalityLevel, OpponentType,
    StrategicReasoningEvaluator, RecursiveSelfModeling
)

__all__ = [
    'MetaCognitiveAgent',
    'MetaCognitiveLayer', 
    'CognitiveState',
    'RecursiveImprovementExperiment',
    'run_recursive_experiment',
    'MetaCognitionLens',
    'InternalState',
    'MetaCognitionLensEvaluator',
    'StepwiseStateAggregator',
    'IntrinsicRewardCalculator',
    'MetaCognitionMetrics',
    'GuessTwoThirdsGame',
    'RationalityLevel',
    'OpponentType',
    'StrategicReasoningEvaluator',
    'RecursiveSelfModeling'
]
