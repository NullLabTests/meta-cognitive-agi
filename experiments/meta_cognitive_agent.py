"""
Meta-Cognitive Agent Implementation
A minimal AI system with self-reflection capabilities using only NumPy
Enhanced with features from arxiv research on meta-cognition and self-awareness
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import json

# Import new meta-cognition features
from .meta_cognition_lenses import (
    MetaCognitionLens, InternalState, MetaCognitionLensEvaluator,
    StepwiseStateAggregator, IntrinsicRewardCalculator, MetaCognitionMetrics
)
from .game_theory_self_awareness import (
    GuessTwoThirdsGame, RationalityLevel, OpponentType,
    StrategicReasoningEvaluator, RecursiveSelfModeling
)


@dataclass
class CognitiveState:
    """Internal state representation of the cognitive process"""
    system_1_output: np.ndarray  # Fast, intuitive reasoning
    system_2_output: np.ndarray  # Slow, deliberate reasoning
    meta_evaluation: np.ndarray  # Self-evaluation of reasoning
    confidence: float  # Confidence in current reasoning
    cognitive_load: float  # Current cognitive resource usage


class MetaCognitiveLayer:
    """
    Implements meta-cognitive self-reflection through dual-process architecture
    System 1: Fast, pattern-based reasoning (low compute)
    System 2: Deliberate, analytical reasoning (high compute)
    Meta-controller: Decides when to engage each system
    """
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 32, meta_dim: int = 16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.meta_dim = meta_dim
        
        # Initialize weights with Xavier/Glorot-like initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        
        # Output projection to match input dimension
        self.W_out = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_out = np.zeros(input_dim)
        
        # Meta-cognitive weights (initialize with correct shape)
        combined_dim = hidden_dim * 2
        self.W_meta = np.random.randn(combined_dim, meta_dim) * np.sqrt(2.0 / combined_dim)
        self.b_meta = np.zeros(meta_dim)
        
        # Self-model weights
        self.W_self = np.random.randn(meta_dim, hidden_dim) * np.sqrt(2.0 / meta_dim)
        self.b_self = np.zeros(hidden_dim)
        
        # System selection weights
        self.W_select = np.random.randn(meta_dim, 2) * np.sqrt(2.0 / meta_dim)
        self.b_select = np.zeros(2)
        
        # Training history
        self.training_history = []
        self.self_awareness_history = []
        
        # Enhanced meta-cognition features (from arxiv research)
        self.lens_evaluator = MetaCognitionLensEvaluator()
        self.reward_calculator = IntrinsicRewardCalculator()
        self.state_aggregator = StepwiseStateAggregator()
        self.metrics_calculator = MetaCognitionMetrics()
        
        # Game theory self-awareness
        self.game_theory = GuessTwoThirdsGame()
        self.strategic_evaluator = StrategicReasoningEvaluator()
        self.recursive_modeling = RecursiveSelfModeling()
        
        # Track internal states for meta-cognition evaluation
        self.internal_states_history = []
        
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def system_1_forward(self, x: np.ndarray) -> np.ndarray:
        """Fast, intuitive reasoning (single layer)"""
        h = self.relu(np.dot(x, self.W1) + self.b1)
        out = np.dot(h, self.W_out) + self.b_out
        return out
    
    def system_2_forward(self, x: np.ndarray) -> np.ndarray:
        """Slow, deliberate reasoning (two layers)"""
        h1 = self.relu(np.dot(x, self.W1) + self.b1)
        h2 = self.relu(np.dot(h1, self.W2) + self.b2)
        out = np.dot(h2, self.W_out) + self.b_out
        return out
    
    def meta_cognitive_forward(self, s1: np.ndarray, s2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Meta-cognitive evaluation and system selection"""
        # Combine both system outputs
        combined = np.concatenate([s1, s2], axis=-1)
        
        # Flatten if needed for matrix multiplication
        if combined.ndim > 2:
            combined = combined.reshape(combined.shape[0], -1)
        
        # Meta-cognitive evaluation
        meta = self.relu(np.dot(combined, self.W_meta) + self.b_meta)
        
        # Self-model prediction
        self_prediction = self.sigmoid(np.dot(meta, self.W_self) + self.b_self)
        
        # System selection (soft decision)
        selection_logits = np.dot(meta, self.W_select) + self.b_select
        selection_probs = self.softmax(selection_logits)
        
        return meta, selection_probs
    
    def forward(self, x: np.ndarray, force_system: Optional[int] = None) -> CognitiveState:
        """Forward pass with meta-cognitive control"""
        # Get outputs from both systems
        s1 = self.system_1_forward(x)
        s2 = self.system_2_forward(x)
        
        # Meta-cognitive evaluation
        meta, selection_probs = self.meta_cognitive_forward(s1, s2)
        
        # Select which system to use (or blend)
        if force_system is not None:
            if force_system == 1:
                output = s1
                confidence = selection_probs[0, 0]
            else:
                output = s2
                confidence = selection_probs[0, 1]
        else:
            # Weighted combination based on meta-cognitive evaluation
            output = selection_probs[0, 0] * s1 + selection_probs[0, 1] * s2
            confidence = np.max(selection_probs)
        
        # Calculate cognitive load (higher for system 2)
        cognitive_load = 0.3 * selection_probs[0, 0] + 1.0 * selection_probs[0, 1]
        
        return CognitiveState(
            system_1_output=s1,
            system_2_output=s2,
            meta_evaluation=meta,
            confidence=confidence,
            cognitive_load=cognitive_load
        )
    
    def self_evaluate(self, state: CognitiveState, target: np.ndarray) -> Dict[str, float]:
        """Evaluate own reasoning process"""
        # Compare output to target
        output = state.system_1_output * 0.5 + state.system_2_output * 0.5
        error = np.mean((output - target) ** 2)
        
        # Self-awareness: measure how well the system understands its own cognitive state
        # This is measured by the consistency between cognitive load and actual performance
        # Use sigmoid to ensure it stays in [0, 1] range
        raw_self_awareness = state.confidence * np.exp(-error)
        self_awareness = float(np.clip(raw_self_awareness, 0.0, 1.0))
        
        # Meta-reasoning: consistency of system selection with task complexity
        # System 2 (slow) should be used more for complex tasks
        meta_reasoning = float(np.clip(state.confidence, 0.0, 1.0))
        
        # Cognitive efficiency: performance per unit cognitive load
        # Use sigmoid to ensure positive values
        raw_efficiency = (1.0 / (1.0 + error)) / (state.cognitive_load + 0.1)
        cognitive_efficiency = float(np.clip(raw_efficiency, 0.0, 10.0))
        
        return {
            'error': float(error),
            'self_awareness': self_awareness,
            'meta_reasoning': meta_reasoning,
            'cognitive_efficiency': cognitive_efficiency
        }
    
    def identify_bottlenecks(self, evaluation: Dict[str, float]) -> List[str]:
        """Identify cognitive bottlenecks based on self-evaluation"""
        bottlenecks = []
        
        if evaluation['self_awareness'] < 0.7:
            bottlenecks.append('self_model_accuracy')
        
        if evaluation['meta_reasoning'] < 0.6:
            bottlenecks.append('system_selection')
        
        if evaluation['cognitive_efficiency'] < 0.5:
            bottlenecks.append('resource_allocation')
        
        if evaluation['error'] > 0.3:
            bottlenecks.append('core_reasoning')
        
        return bottlenecks
    
    def generate_improvements(self, bottlenecks: List[str]) -> Dict[str, np.ndarray]:
        """Generate targeted improvements for identified bottlenecks"""
        improvements = {}
        
        for bottleneck in bottlenecks:
            if bottleneck == 'self_model_accuracy':
                # Improve self-model weights
                improvements['W_self'] = self.W_self * 1.05  # 5% increase
                improvements['b_self'] = self.b_self * 0.95
                
            elif bottleneck == 'system_selection':
                # Improve selection weights
                improvements['W_select'] = self.W_select * 1.03
                improvements['b_select'] = self.b_select * 0.97
                
            elif bottleneck == 'resource_allocation':
                # Improve meta-cognitive weights
                improvements['W_meta'] = self.W_meta * 1.02
                improvements['b_meta'] = self.b_meta * 0.98
                
            elif bottleneck == 'core_reasoning':
                # Improve core reasoning weights
                improvements['W1'] = self.W1 * 1.04
                improvements['W2'] = self.W2 * 1.04
                improvements['W_out'] = self.W_out * 1.03
                improvements['b_out'] = self.b_out * 0.97
        
        return improvements
    
    def apply_improvements(self, improvements: Dict[str, np.ndarray]):
        """Apply the generated improvements"""
        for key, value in improvements.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def train_step(self, x: np.ndarray, target: np.ndarray, learning_rate: float = 0.001) -> Dict[str, float]:
        """Single training step with backpropagation"""
        # Forward pass
        state = self.forward(x)
        
        # Compute gradients (simplified with clipping)
        output = state.system_1_output * 0.5 + state.system_2_output * 0.5
        error = output - target
        
        # Gradient for W1 (simplified with normalization)
        grad_W1 = np.dot(x.T, error) / (x.shape[0] + 1e-8)
        grad_W_out = np.dot(state.system_2_output.T, error) / (x.shape[0] + 1e-8)
        
        # Clip gradients to prevent explosion
        grad_W1 = np.clip(grad_W1, -1.0, 1.0)
        grad_W_out = np.clip(grad_W_out, -1.0, 1.0)
        
        # Update weights with smaller learning rate
        self.W1 -= learning_rate * grad_W1
        self.W_out -= learning_rate * grad_W_out
        
        # Normalize weights periodically
        if np.random.random() < 0.1:  # 10% chance to normalize
            self.W1 = self.W1 / (np.linalg.norm(self.W1) + 1e-8)
            self.W_out = self.W_out / (np.linalg.norm(self.W_out) + 1e-8)
        
        # Self-evaluation
        metrics = self.self_evaluate(state, target)
        
        # Store history
        self.training_history.append(metrics)
        self.self_awareness_history.append(metrics['self_awareness'])
        
        return metrics
    
    def extract_internal_state(self, x: np.ndarray) -> InternalState:
        """
        Extract internal state for meta-cognitive evaluation
        Based on methodology from "Large Language Models Have Intrinsic Meta-Cognition"
        """
        # Forward pass to get internal representations
        h1 = self.relu(np.dot(x, self.W1) + self.b1)
        h2 = self.relu(np.dot(h1, self.W2) + self.b2)
        
        # Use hidden states as internal representation
        hidden_states = h2  # Shape: (batch, hidden_dim)
        
        # Generate logits (simulated output layer)
        logits = np.dot(h2, self.W_out) + self.b_out
        
        # Convert to probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Calculate confidence from system selection
        meta = self.relu(np.dot(np.concatenate([h1, h2], axis=-1), self.W_meta) + self.b_meta)
        selection_logits = np.dot(meta, self.W_select) + self.b_select
        selection_probs = self.softmax(selection_logits)
        confidence = float(np.max(selection_probs))
        
        return InternalState(
            hidden_states=hidden_states,
            logits=logits,
            probabilities=probabilities,
            confidence=confidence
        )
    
    def evaluate_with_meta_cognition_lenses(self, x: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """
        Evaluate using multiple meta-cognition lenses
        Returns confidence scores from different lenses
        """
        state = self.extract_internal_state(x)
        
        lens_results = {}
        for lens in MetaCognitionLens:
            lens_results[lens.value] = self.lens_evaluator.evaluate_with_lens(state, lens)
        
        # Store internal state for tracking
        self.internal_states_history.append(state)
        
        return lens_results
    
    def calculate_intrinsic_reward(self, x: np.ndarray, lens: MetaCognitionLens) -> float:
        """
        Calculate intrinsic reward using a specific meta-cognition lens
        """
        state = self.extract_internal_state(x)
        return self.lens_evaluator.evaluate_with_lens(state, lens)
    
    def evaluate_game_theory_self_awareness(self, guess: float, opponent_type: OpponentType) -> Dict[str, float]:
        """
        Evaluate self-awareness using game theory paradigm
        Based on "LLs Position Themselves as More Rational Than Humans"
        """
        self_awareness = self.game_theory.calculate_self_awareness_score(guess, opponent_type)
        rationality = self.game_theory.determine_rationality_level(guess)
        
        return {
            'self_awareness_score': self_awareness,
            'rationality_level': rationality.value,
            'guess': guess
        }


class MetaCognitiveAgent:
    """
    Complete meta-cognitive agent with training and evaluation capabilities
    """
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 32, meta_dim: int = 16):
        self.meta_layer = MetaCognitiveLayer(input_dim, hidden_dim, meta_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.meta_dim = meta_dim
        
        # Performance tracking
        self.performance_history = []
        self.refinement_cycles = 0
        
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
              refinement_cycles: int = 5) -> Dict[str, List[float]]:
        """Train the agent with periodic self-refinement"""
        history = {
            'error': [],
            'self_awareness': [],
            'meta_reasoning': [],
            'cognitive_efficiency': []
        }
        
        for epoch in range(epochs):
            epoch_metrics = []
            
            for i in range(len(X)):
                metrics = self.meta_layer.train_step(X[i:i+1], y[i:i+1])
                epoch_metrics.append(metrics)
            
            # Average metrics for epoch
            avg_metrics = {
                key: np.mean([m[key] for m in epoch_metrics])
                for key in epoch_metrics[0].keys()
            }
            
            for key, value in avg_metrics.items():
                history[key].append(value)
            
            # Periodic self-refinement
            if epoch % (epochs // refinement_cycles) == 0 and epoch > 0:
                self.self_refine(avg_metrics)
                self.refinement_cycles += 1
        
        return history
    
    def self_refine(self, metrics: Dict[str, float]):
        """Perform self-reflection and improvement"""
        bottlenecks = self.meta_layer.identify_bottlenecks(metrics)
        
        if bottlenecks:
            improvements = self.meta_layer.generate_improvements(bottlenecks)
            self.meta_layer.apply_improvements(improvements)
    
    def evaluate_self_awareness(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate self-awareness on test data"""
        self_awareness_scores = []
        
        for i in range(len(X_test)):
            state = self.meta_layer.forward(X_test[i:i+1])
            metrics = self.meta_layer.self_evaluate(state, y_test[i:i+1])
            self_awareness_scores.append(metrics['self_awareness'])
        
        return np.mean(self_awareness_scores)
    
    def evaluate_meta_reasoning(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate meta-reasoning capability"""
        meta_reasoning_scores = []
        
        for i in range(len(X_test)):
            state = self.meta_layer.forward(X_test[i:i+1])
            metrics = self.meta_layer.self_evaluate(state, y_test[i:i+1])
            meta_reasoning_scores.append(metrics['meta_reasoning'])
        
        return np.mean(meta_reasoning_scores)
    
    def evaluate_with_lenses(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate using all meta-cognition lenses
        Returns nested dict of lens results for each sample
        """
        all_lens_results = {}
        
        for i in range(len(X_test)):
            lens_results = self.meta_layer.evaluate_with_meta_cognition_lenses(X_test[i:i+1], y_test[i:i+1])
            all_lens_results[f'sample_{i}'] = lens_results
        
        # Aggregate across samples
        aggregated = {}
        for lens in MetaCognitionLens:
            values = [all_lens_results[f'sample_{i}'][lens.value] for i in range(len(X_test))]
            aggregated[lens.value] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return aggregated
    
    def evaluate_game_theory_awareness(self, guesses: List[float], 
                                     opponent_type: OpponentType) -> Dict[str, float]:
        """
        Evaluate self-awareness using game theory paradigm
        """
        profile = self.meta_layer.strategic_evaluator.evaluate_agent_strategic_reasoning(
            guesses, opponent_type
        )
        return profile
    
    def calculate_meta_cognition_metrics(self, confidence_scores: List[float], 
                                       correctness_labels: List[int]) -> Dict[str, float]:
        """
        Calculate meta-cognition evaluation metrics (AUPR, AUROC, FPR95)
        """
        aupr = self.meta_layer.metrics_calculator.calculate_aupr(confidence_scores, correctness_labels)
        auroc = self.meta_layer.metrics_calculator.calculate_auroc(confidence_scores, correctness_labels)
        fpr95 = self.meta_layer.metrics_calculator.calculate_fpr95(confidence_scores, correctness_labels)
        
        return {
            'aupr': aupr,
            'auroc': auroc,
            'fpr95': fpr95
        }
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of agent performance"""
        if not self.meta_layer.training_history:
            return {}
        
        recent_metrics = self.meta_layer.training_history[-10:]
        
        return {
            'avg_error': np.mean([m['error'] for m in recent_metrics]),
            'avg_self_awareness': np.mean([m['self_awareness'] for m in recent_metrics]),
            'avg_meta_reasoning': np.mean([m['meta_reasoning'] for m in recent_metrics]),
            'avg_cognitive_efficiency': np.mean([m['cognitive_efficiency'] for m in recent_metrics]),
            'refinement_cycles': self.refinement_cycles
        }
    
    def save_state(self, filepath: str):
        """Save agent state to file"""
        state = {
            'W1': self.meta_layer.W1.tolist(),
            'b1': self.meta_layer.b1.tolist(),
            'W2': self.meta_layer.W2.tolist(),
            'b2': self.meta_layer.b2.tolist(),
            'W_out': self.meta_layer.W_out.tolist(),
            'b_out': self.meta_layer.b_out.tolist(),
            'W_meta': self.meta_layer.W_meta.tolist(),
            'b_meta': self.meta_layer.b_meta.tolist(),
            'W_self': self.meta_layer.W_self.tolist(),
            'b_self': self.meta_layer.b_self.tolist(),
            'W_select': self.meta_layer.W_select.tolist(),
            'b_select': self.meta_layer.b_select.tolist(),
            'training_history': self.meta_layer.training_history,
            'refinement_cycles': self.refinement_cycles,
            'internal_states_history': len(self.meta_layer.internal_states_history)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load agent state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.meta_layer.W1 = np.array(state['W1'])
        self.meta_layer.b1 = np.array(state['b1'])
        self.meta_layer.W2 = np.array(state['W2'])
        self.meta_layer.b2 = np.array(state['b2'])
        self.meta_layer.W_out = np.array(state['W_out'])
        self.meta_layer.b_out = np.array(state['b_out'])
        self.meta_layer.W_meta = np.array(state['W_meta'])
        self.meta_layer.b_meta = np.array(state['b_meta'])
        self.meta_layer.W_self = np.array(state['W_self'])
        self.meta_layer.b_self = np.array(state['b_self'])
        self.meta_layer.W_select = np.array(state['W_select'])
        self.meta_layer.b_select = np.array(state['b_select'])
        self.meta_layer.training_history = state['training_history']
        self.refinement_cycles = state['refinement_cycles']


def generate_synthetic_arc_tasks(n_tasks: int = 100, input_dim: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic ARC-like reasoning tasks"""
    np.random.seed(42)
    
    X = np.random.randn(n_tasks, input_dim)
    y = np.zeros((n_tasks, input_dim))
    
    for i in range(n_tasks):
        # Create various reasoning patterns
        pattern_type = i % 5
        
        if pattern_type == 0:  # Pattern completion
            y[i] = X[i] * 0.5 + np.roll(X[i], 1) * 0.5
        elif pattern_type == 1:  # Symmetry detection
            y[i] = (X[i] + X[i][::-1]) / 2
        elif pattern_type == 2:  # Transformation
            y[i] = X[i] * 2 - 1
        elif pattern_type == 3:  # Composition
            y[i] = np.sin(X[i]) * np.cos(X[i])
        else:  # Abstraction
            y[i] = np.sign(X[i]) * np.abs(X[i]) ** 0.5
    
    return X, y


if __name__ == "__main__":
    print("Initializing Meta-Cognitive Agent...")
    
    # Generate synthetic tasks
    print("Generating synthetic ARC-like tasks...")
    X_train, y_train = generate_synthetic_arc_tasks(n_tasks=500)
    X_test, y_test = generate_synthetic_arc_tasks(n_tasks=100)
    
    # Initialize agent
    agent = MetaCognitiveAgent(input_dim=32, hidden_dim=32, meta_dim=16)
    
    # Train agent
    print("Training meta-cognitive agent...")
    history = agent.train(X_train, y_train, epochs=100, refinement_cycles=5)
    
    # Evaluate
    print("\nEvaluating agent performance...")
    self_awareness = agent.evaluate_self_awareness(X_test, y_test)
    meta_reasoning = agent.evaluate_meta_reasoning(X_test, y_test)
    
    print(f"\nSelf-Awareness Score: {self_awareness:.3f}")
    print(f"Meta-Reasoning Score: {meta_reasoning:.3f}")
    
    # Performance summary
    summary = agent.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"Average Error: {summary['avg_error']:.4f}")
    print(f"Average Self-Awareness: {summary['avg_self_awareness']:.4f}")
    print(f"Average Meta-Reasoning: {summary['avg_meta_reasoning']:.4f}")
    print(f"Average Cognitive Efficiency: {summary['avg_cognitive_efficiency']:.4f}")
    print(f"Refinement Cycles: {summary['refinement_cycles']}")
    
    # Save agent
    agent.save_state("meta_cognitive_agent_state.json")
    print("\nAgent state saved to meta_cognitive_agent_state.json")
