"""
Recursive Self-Improvement Simulation
Studies how minimal AI systems can autonomously enhance their cognitive capabilities
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.meta_cognitive_agent import MetaCognitiveAgent, generate_synthetic_arc_tasks


@dataclass
class ImprovementCycle:
    """Results from a single improvement cycle"""
    cycle_number: int
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    bottlenecks_identified: List[str]
    improvements_applied: List[str]
    improvement_gain: float


class RecursiveImprovementExperiment:
    """
    Simulates recursive self-improvement in meta-cognitive agents
    """
    
    def __init__(self, agent: MetaCognitiveAgent, max_cycles: int = 10):
        self.agent = agent
        self.max_cycles = max_cycles
        self.improvement_history = []
        self.converged = False
        
    def run_cycle(self, X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray) -> ImprovementCycle:
        """Run a single improvement cycle"""
        cycle_num = len(self.improvement_history) + 1
        
        # Evaluate performance before improvement
        perf_before = self._evaluate_agent(X_val, y_val)
        
        # Identify bottlenecks
        bottlenecks = self.agent.meta_layer.identify_bottlenecks(perf_before)
        
        # Generate and apply improvements
        if bottlenecks:
            improvements = self.agent.meta_layer.generate_improvements(bottlenecks)
            self.agent.meta_layer.apply_improvements(improvements)
        else:
            improvements = {}
        
        # Retrain after improvements
        self.agent.train(X_train, y_train, epochs=20, refinement_cycles=1)
        
        # Evaluate performance after improvement
        perf_after = self._evaluate_agent(X_val, y_val)
        
        # Calculate improvement gain
        improvement_gain = perf_after['self_awareness'] - perf_before['self_awareness']
        
        # Check for convergence
        if improvement_gain < 0.01 or not bottlenecks:
            self.converged = True
        
        cycle = ImprovementCycle(
            cycle_number=cycle_num,
            performance_before=perf_before,
            performance_after=perf_after,
            bottlenecks_identified=bottlenecks,
            improvements_applied=list(improvements.keys()),
            improvement_gain=improvement_gain
        )
        
        self.improvement_history.append(cycle)
        return cycle
    
    def _evaluate_agent(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate agent on given data"""
        self_awareness = self.agent.evaluate_self_awareness(X, y)
        meta_reasoning = self.agent.evaluate_meta_reasoning(X, y)
        
        # Additional metrics
        summary = self.agent.get_performance_summary()
        
        return {
            'self_awareness': self_awareness,
            'meta_reasoning': meta_reasoning,
            'error': summary.get('avg_error', 0.5),
            'cognitive_efficiency': summary.get('avg_cognitive_efficiency', 0.5)
        }
    
    def run_full_experiment(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> List[ImprovementCycle]:
        """Run the complete recursive improvement experiment"""
        print(f"Starting recursive improvement experiment (max {self.max_cycles} cycles)...")
        
        for cycle in range(self.max_cycles):
            print(f"\n--- Cycle {cycle + 1} ---")
            
            improvement_cycle = self.run_cycle(X_train, y_train, X_val, y_val)
            
            print(f"Bottlenecks identified: {improvement_cycle.bottlenecks_identified}")
            print(f"Improvements applied: {improvement_cycle.improvements_applied}")
            print(f"Self-awareness: {improvement_cycle.performance_before['self_awareness']:.4f} -> "
                  f"{improvement_cycle.performance_after['self_awareness']:.4f}")
            print(f"Improvement gain: {improvement_cycle.improvement_gain:.4f}")
            
            if self.converged:
                print(f"\nConverged after {cycle + 1} cycles")
                break
        
        return self.improvement_history
    
    def get_improvement_trajectory(self) -> Dict[str, List[float]]:
        """Extract improvement trajectory for visualization"""
        trajectory = {
            'cycle_numbers': [],
            'self_awareness': [],
            'meta_reasoning': [],
            'error': [],
            'cognitive_efficiency': [],
            'improvement_gain': []
        }
        
        for cycle in self.improvement_history:
            trajectory['cycle_numbers'].append(cycle.cycle_number)
            trajectory['self_awareness'].append(cycle.performance_after['self_awareness'])
            trajectory['meta_reasoning'].append(cycle.performance_after['meta_reasoning'])
            trajectory['error'].append(cycle.performance_after['error'])
            trajectory['cognitive_efficiency'].append(cycle.performance_after['cognitive_efficiency'])
            trajectory['improvement_gain'].append(cycle.improvement_gain)
        
        return trajectory
    
    def save_results(self, filepath: str):
        """Save experiment results to file"""
        results = {
            'max_cycles': self.max_cycles,
            'converged': self.converged,
            'total_cycles': len(self.improvement_history),
            'improvement_history': [
                {
                    'cycle_number': cycle.cycle_number,
                    'performance_before': cycle.performance_before,
                    'performance_after': cycle.performance_after,
                    'bottlenecks_identified': cycle.bottlenecks_identified,
                    'improvements_applied': cycle.improvements_applied,
                    'improvement_gain': cycle.improvement_gain
                }
                for cycle in self.improvement_history
            ],
            'trajectory': self.get_improvement_trajectory()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filepath}")


def run_recursive_experiment(initial_agent: str = "meta_cognitive_50k",
                           max_cycles: int = 10,
                           evaluation_tasks: List[str] = ["arc_agi_1"]) -> Dict:
    """
    Run recursive improvement experiment with specified parameters
    """
    print("=" * 60)
    print("RECURSIVE SELF-IMPROVEMENT EXPERIMENT")
    print("=" * 60)
    
    # Generate data
    print("\nGenerating training and validation data...")
    X_train, y_train = generate_synthetic_arc_tasks(n_tasks=500)
    X_val, y_val = generate_synthetic_arc_tasks(n_tasks=100)
    
    # Initialize agent
    print(f"Initializing {initial_agent} agent...")
    agent = MetaCognitiveAgent(input_dim=32, hidden_dim=64, meta_dim=32)
    
    # Initial training
    print("Performing initial training...")
    agent.train(X_train, y_train, epochs=50, refinement_cycles=2)
    
    # Run recursive improvement
    experiment = RecursiveImprovementExperiment(agent, max_cycles=max_cycles)
    results = experiment.run_full_experiment(X_train, y_train, X_val, y_val)
    
    # Save results
    experiment.save_results("recursive_improvement_results.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    if results:
        final_cycle = results[-1]
        print(f"\nFinal Performance:")
        print(f"  Self-Awareness: {final_cycle.performance_after['self_awareness']:.4f}")
        print(f"  Meta-Reasoning: {final_cycle.performance_after['meta_reasoning']:.4f}")
        print(f"  Error: {final_cycle.performance_after['error']:.4f}")
        print(f"  Cognitive Efficiency: {final_cycle.performance_after['cognitive_efficiency']:.4f}")
        
        print(f"\nImprovement Statistics:")
        initial_sa = results[0].performance_before['self_awareness']
        final_sa = final_cycle.performance_after['self_awareness']
        total_improvement = final_sa - initial_sa
        print(f"  Total Self-Awareness Improvement: {total_improvement:.4f}")
        print(f"  Relative Improvement: {(total_improvement / initial_sa * 100):.2f}%")
        print(f"  Cycles to Converge: {len(results)}")
    
    return {
        'agent': agent,
        'experiment': experiment,
        'results': results
    }


if __name__ == "__main__":
    # Run the experiment
    result = run_recursive_experiment(
        initial_agent="meta_cognitive_50k",
        max_cycles=10,
        evaluation_tasks=["arc_agi_1", "arc_agi_2"]
    )
    
    print("\nExperiment completed successfully!")
