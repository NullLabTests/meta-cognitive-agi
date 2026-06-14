"""
Meta-Cognitive AGI Experiments
"""

from .meta_cognitive_agent import MetaCognitiveAgent, MetaCognitiveLayer, CognitiveState
from .recursive_improvement import RecursiveImprovementExperiment, run_recursive_experiment

__all__ = [
    'MetaCognitiveAgent',
    'MetaCognitiveLayer', 
    'CognitiveState',
    'RecursiveImprovementExperiment',
    'run_recursive_experiment'
]
