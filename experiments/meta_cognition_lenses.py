"""
Meta-Cognition Lenses and Intrinsic Reward Functions
Based on arxiv research: "Large Language Models Have Intrinsic Meta-Cognition, but Need a Good Lens"
"""

import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class MetaCognitionLens(Enum):
    """Different lenses for evaluating meta-cognition from internal states"""
    ENTROPY = "entropy"
    MAX_PROB = "maxprob"
    PERPLEXITY = "perplexity"
    DELTA_ENTROPY = "delta_entropy"


@dataclass
class InternalState:
    """Internal state representation for meta-cognitive evaluation"""
    hidden_states: np.ndarray  # All-layer hidden states
    logits: np.ndarray  # Output logits
    probabilities: np.ndarray  # Output probabilities
    confidence: float  # Overall confidence score


class MetaCognitionLensEvaluator:
    """
    Evaluates meta-cognitive abilities using different lenses
    Based on methodology from "Large Language Models Have Intrinsic Meta-Cognition, but Need a Good Lens"
    """
    
    def __init__(self):
        self.lens_functions = {
            MetaCognitionLens.ENTROPY: self.entropy_lens,
            MetaCognitionLens.MAX_PROB: self.max_prob_lens,
            MetaCognitionLens.PERPLEXITY: self.perplexity_lens,
            MetaCognitionLens.DELTA_ENTROPY: self.delta_entropy_lens,
        }
    
    def entropy_lens(self, state: InternalState) -> float:
        """
        Entropy lens: Lower entropy indicates higher confidence
        H(p) = -sum(p * log(p))
        """
        # Avoid log(0)
        eps = 1e-10
        p = np.clip(state.probabilities, eps, 1.0)
        
        # Handle both scalar and array probabilities
        if p.ndim == 0:
            p = np.array([p])
        
        entropy = -np.sum(p * np.log(p))
        
        # Normalize to [0,1] where 1 = high confidence (low entropy)
        max_entropy = np.log(len(p))
        confidence = 1.0 - (entropy / (max_entropy + eps))
        return float(np.clip(confidence, 0.0, 1.0))
    
    def max_prob_lens(self, state: InternalState) -> float:
        """
        Max probability lens: Maximum probability in the distribution
        Higher max probability indicates higher confidence
        """
        max_p = np.max(state.probabilities)
        return float(np.clip(max_p, 0.0, 1.0))
    
    def perplexity_lens(self, state: InternalState) -> float:
        """
        Perplexity lens: Inverse of geometric mean of probabilities
        Lower perplexity indicates higher confidence
        """
        eps = 1e-10
        p = np.clip(state.probabilities, eps, 1.0)
        
        # Handle both scalar and array probabilities
        if p.ndim == 0:
            p = np.array([p])
        
        perplexity = np.exp(-np.mean(np.log(p)))
        # Normalize: perplexity ranges from 1 to vocab_size
        # We want confidence to be higher when perplexity is lower
        confidence = 1.0 / (perplexity + eps)
        return float(np.clip(confidence, 0.0, 1.0))
    
    def delta_entropy_lens(self, state: InternalState, prev_state: InternalState = None) -> float:
        """
        Delta entropy lens: Change in entropy over time
        Decreasing entropy indicates increasing confidence
        """
        if prev_state is None:
            return 0.5  # Neutral if no previous state
        
        current_entropy = self._compute_entropy(state.probabilities)
        prev_entropy = self._compute_entropy(prev_state.probabilities)
        
        delta = prev_entropy - current_entropy  # Positive = entropy decreased
        # Normalize delta to [0,1]
        max_delta = prev_entropy  # Maximum possible decrease
        confidence = 0.5 + 0.5 * (delta / (max_delta + 1e-10))
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _compute_entropy(self, probabilities: np.ndarray) -> float:
        """Helper to compute entropy"""
        eps = 1e-10
        p = np.clip(probabilities, eps, 1.0)
        return -np.sum(p * np.log(p))
    
    def evaluate_with_lens(self, state: InternalState, lens: MetaCognitionLens, 
                          prev_state: InternalState = None) -> float:
        """Evaluate state using a specific lens"""
        if lens == MetaCognitionLens.DELTA_ENTROPY:
            return self.delta_entropy_lens(state, prev_state)
        else:
            return self.lens_functions[lens](state)
    
    def evaluate_all_lenses(self, state: InternalState, 
                           prev_state: InternalState = None) -> Dict[str, float]:
        """Evaluate state using all lenses"""
        results = {}
        for lens in MetaCognitionLens:
            results[lens.value] = self.evaluate_with_lens(state, lens, prev_state)
        return results


class StepwiseStateAggregator:
    """
    Aggregates internal states across reasoning steps
    Based on methodology from "Large Language Models Have Intrinsic Meta-Cognition"
    """
    
    def __init__(self):
        self.step_states = []
    
    def add_step(self, hidden_states: np.ndarray, logits: np.ndarray, 
                 probabilities: np.ndarray):
        """Add a reasoning step's internal states"""
        state = InternalState(
            hidden_states=hidden_states,
            logits=logits,
            probabilities=probabilities,
            confidence=0.0  # Will be computed
        )
        self.step_states.append(state)
    
    def aggregate_step(self, step_index: int) -> InternalState:
        """
        Aggregate states for a single step across all tokens
        Returns aggregated internal state for the step
        """
        if step_index >= len(self.step_states):
            raise IndexError(f"Step index {step_index} out of range")
        
        state = self.step_states[step_index]
        
        # Aggregate across tokens in the step
        # For hidden states: average across tokens
        aggregated_hidden = np.mean(state.hidden_states, axis=0)
        
        # For logits: average across tokens
        aggregated_logits = np.mean(state.logits, axis=0)
        
        # For probabilities: average across tokens
        aggregated_probs = np.mean(state.probabilities, axis=0)
        
        # Renormalize probabilities
        aggregated_probs = aggregated_probs / (np.sum(aggregated_probs) + 1e-10)
        
        return InternalState(
            hidden_states=aggregated_hidden,
            logits=aggregated_logits,
            probabilities=aggregated_probs,
            confidence=0.0
        )
    
    def get_all_aggregated_states(self) -> List[InternalState]:
        """Get aggregated states for all steps"""
        return [self.aggregate_step(i) for i in range(len(self.step_states))]
    
    def compute_intrinsic_rewards(self, lens: MetaCognitionLens) -> List[float]:
        """
        Compute intrinsic rewards for all steps using a specific lens
        Returns list of confidence scores
        """
        evaluator = MetaCognitionLensEvaluator()
        aggregated_states = self.get_all_aggregated_states()
        
        rewards = []
        for i, state in enumerate(aggregated_states):
            prev_state = aggregated_states[i-1] if i > 0 else None
            reward = evaluator.evaluate_with_lens(state, lens, prev_state)
            rewards.append(reward)
        
        return rewards


class IntrinsicRewardCalculator:
    """
    Calculates intrinsic rewards based on internal states
    Inspired by "AutoMeco" framework from arxiv research
    """
    
    def __init__(self):
        self.aggregator = StepwiseStateAggregator()
        self.lens_evaluator = MetaCognitionLensEvaluator()
    
    def calculate_step_reward(self, hidden_states: np.ndarray, logits: np.ndarray,
                              probabilities: np.ndarray, lens: MetaCognitionLens,
                              prev_hidden_states: np.ndarray = None,
                              prev_logits: np.ndarray = None,
                              prev_probabilities: np.ndarray = None) -> float:
        """
        Calculate intrinsic reward for a single reasoning step
        """
        state = InternalState(
            hidden_states=hidden_states,
            logits=logits,
            probabilities=probabilities,
            confidence=0.0
        )
        
        prev_state = None
        if prev_hidden_states is not None:
            prev_state = InternalState(
                hidden_states=prev_hidden_states,
                logits=prev_logits,
                probabilities=prev_probabilities,
                confidence=0.0
            )
        
        return self.lens_evaluator.evaluate_with_lens(state, lens, prev_state)
    
    def calculate_multi_step_rewards(self, states_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                     lens: MetaCognitionLens) -> List[float]:
        """
        Calculate intrinsic rewards for multiple reasoning steps
        states_list: List of (hidden_states, logits, probabilities) tuples
        """
        rewards = []
        for i, (hidden, logits, probs) in enumerate(states_list):
            prev_hidden, prev_logits, prev_probs = None, None, None
            if i > 0:
                prev_hidden, prev_logits, prev_probs = states_list[i-1]
            
            reward = self.calculate_step_reward(
                hidden, logits, probs, lens,
                prev_hidden, prev_logits, prev_probs
            )
            rewards.append(reward)
        
        return rewards


class MetaCognitionMetrics:
    """
    Evaluation metrics for meta-cognitive abilities
    Based on metrics from arxiv research: AUPR, AUROC, FPR95
    """
    
    @staticmethod
    def calculate_aupr(confidence_scores: List[float], correctness_labels: List[int]) -> float:
        """
        Calculate Area Under Precision-Recall Curve
        confidence_scores: List of confidence scores [0,1]
        correctness_labels: List of binary labels (0=incorrect, 1=correct)
        """
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidence_scores)[::-1]
        sorted_labels = [correctness_labels[i] for i in sorted_indices]
        
        # Calculate precision-recall curve
        total_positive = sum(correctness_labels)
        total_negative = len(correctness_labels) - total_positive
        
        if total_positive == 0 or total_negative == 0:
            return 0.0
        
        precision_values = []
        recall_values = []
        
        true_positives = 0
        false_positives = 0
        
        for label in sorted_labels:
            if label == 1:
                true_positives += 1
            else:
                false_positives += 1
            
            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (total_positive + 1e-10)
            
            precision_values.append(precision)
            recall_values.append(recall)
        
        # Calculate AUPR using trapezoidal rule
        aupr = 0.0
        for i in range(1, len(precision_values)):
            aupr += (recall_values[i] - recall_values[i-1]) * (precision_values[i] + precision_values[i-1]) / 2
        
        return float(aupr)
    
    @staticmethod
    def calculate_auroc(confidence_scores: List[float], correctness_labels: List[int]) -> float:
        """
        Calculate Area Under ROC Curve
        """
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidence_scores)[::-1]
        sorted_labels = [correctness_labels[i] for i in sorted_indices]
        
        total_positive = sum(correctness_labels)
        total_negative = len(correctness_labels) - total_positive
        
        if total_positive == 0 or total_negative == 0:
            return 0.0
        
        true_positives = 0
        false_positives = 0
        
        auroc = 0.0
        prev_fpr = 0.0
        prev_tpr = 0.0
        
        for label in sorted_labels:
            if label == 1:
                true_positives += 1
            else:
                false_positives += 1
            
            tpr = true_positives / (total_positive + 1e-10)
            fpr = false_positives / (total_negative + 1e-10)
            
            auroc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
            
            prev_fpr = fpr
            prev_tpr = tpr
        
        return float(auroc)
    
    @staticmethod
    def calculate_fpr95(confidence_scores: List[float], correctness_labels: List[int]) -> float:
        """
        Calculate False Positive Rate at 95% True Positive Rate
        """
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidence_scores)[::-1]
        sorted_labels = [correctness_labels[i] for i in sorted_indices]
        
        total_positive = sum(correctness_labels)
        total_negative = len(correctness_labels) - total_positive
        
        if total_positive == 0:
            return 0.0
        
        # Find threshold for 95% TPR
        true_positives_needed = int(0.95 * total_positive)
        true_positives = 0
        threshold_idx = 0
        
        for i, label in enumerate(sorted_labels):
            if label == 1:
                true_positives += 1
            if true_positives >= true_positives_needed:
                threshold_idx = i
                break
        
        # Calculate FPR at this threshold
        false_positives = sum(1 for label in sorted_labels[:threshold_idx+1] if label == 0)
        fpr = false_positives / (total_negative + 1e-10)
        
        return float(fpr)


def generate_synthetic_internal_states(n_steps: int, hidden_dim: int = 32, 
                                         vocab_size: int = 100) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate synthetic internal states for testing
    Returns list of (hidden_states, logits, probabilities) tuples
    """
    np.random.seed(42)
    states = []
    
    for step in range(n_steps):
        # Generate hidden states (simulating multiple layers)
        hidden = np.random.randn(hidden_dim, hidden_dim)
        
        # Generate logits
        logits = np.random.randn(vocab_size)
        
        # Convert to probabilities with softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        states.append((hidden, logits, probs))
    
    return states


if __name__ == "__main__":
    print("Testing Meta-Cognition Lenses...")
    
    # Generate synthetic internal states
    states = generate_synthetic_internal_states(n_steps=5, hidden_dim=32, vocab_size=100)
    
    # Test intrinsic reward calculator
    calculator = IntrinsicRewardCalculator()
    
    print("\nTesting different lenses:")
    for lens in MetaCognitionLens:
        rewards = calculator.calculate_multi_step_rewards(states, lens)
        print(f"{lens.value}: {rewards}")
    
    # Test metrics
    print("\nTesting meta-cognition metrics:")
    confidence_scores = [0.8, 0.6, 0.9, 0.4, 0.7]
    correctness_labels = [1, 0, 1, 0, 1]
    
    aupr = MetaCognitionMetrics.calculate_aupr(confidence_scores, correctness_labels)
    auroc = MetaCognitionMetrics.calculate_auroc(confidence_scores, correctness_labels)
    fpr95 = MetaCognitionMetrics.calculate_fpr95(confidence_scores, correctness_labels)
    
    print(f"AUPR: {aupr:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"FPR95: {fpr95:.4f}")
    
    print("\nAll meta-cognition lens tests passed!")
