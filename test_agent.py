"""
Test suite for Meta-Cognitive Agent
Verifies that the agent works correctly and produces meaningful results
"""

import numpy as np
from experiments.meta_cognitive_agent import MetaCognitiveAgent, generate_synthetic_arc_tasks


def test_agent_initialization():
    """Test that agent initializes correctly"""
    print("Testing agent initialization...")
    agent = MetaCognitiveAgent(input_dim=32, hidden_dim=32, meta_dim=16)
    
    assert agent.input_dim == 32
    assert agent.hidden_dim == 32
    assert agent.meta_dim == 16
    assert agent.meta_layer is not None
    print("✓ Agent initialization passed")


def test_forward_pass():
    """Test that forward pass works"""
    print("\nTesting forward pass...")
    agent = MetaCognitiveAgent(input_dim=32, hidden_dim=32, meta_dim=16)
    
    x = np.random.randn(1, 32)
    state = agent.meta_layer.forward(x)
    
    assert state.system_1_output.shape == (1, 32)
    assert state.system_2_output.shape == (1, 32)
    assert state.meta_evaluation.shape == (1, 16)
    assert 0 <= state.confidence <= 1
    assert state.cognitive_load > 0
    print("✓ Forward pass passed")


def test_training():
    """Test that training works"""
    print("\nTesting training...")
    agent = MetaCognitiveAgent(input_dim=32, hidden_dim=32, meta_dim=16)
    
    X_train, y_train = generate_synthetic_arc_tasks(n_tasks=50)
    
    history = agent.train(X_train, y_train, epochs=10, refinement_cycles=2)
    
    assert 'error' in history
    assert 'self_awareness' in history
    assert 'meta_reasoning' in history
    assert len(history['error']) == 10
    print("✓ Training passed")


def test_self_evaluation():
    """Test that self-evaluation works"""
    print("\nTesting self-evaluation...")
    agent = MetaCognitiveAgent(input_dim=32, hidden_dim=32, meta_dim=16)
    
    x = np.random.randn(1, 32)
    y = np.random.randn(1, 32)
    
    state = agent.meta_layer.forward(x)
    metrics = agent.meta_layer.self_evaluate(state, y)
    
    assert 'error' in metrics
    assert 'self_awareness' in metrics
    assert 'meta_reasoning' in metrics
    assert 'cognitive_efficiency' in metrics
    assert metrics['error'] >= 0
    assert 0 <= metrics['self_awareness'] <= 1
    print("✓ Self-evaluation passed")


def test_bottleneck_identification():
    """Test that bottleneck identification works"""
    print("\nTesting bottleneck identification...")
    agent = MetaCognitiveAgent(input_dim=32, hidden_dim=32, meta_dim=16)
    
    # Test with poor metrics
    evaluation = {
        'self_awareness': 0.5,
        'meta_reasoning': 0.4,
        'cognitive_efficiency': 0.3,
        'error': 0.5
    }
    
    bottlenecks = agent.meta_layer.identify_bottlenecks(evaluation)
    assert len(bottlenecks) > 0
    print(f"  Identified bottlenecks: {bottlenecks}")
    print("✓ Bottleneck identification passed")


def test_improvement_generation():
    """Test that improvement generation works"""
    print("\nTesting improvement generation...")
    agent = MetaCognitiveAgent(input_dim=32, hidden_dim=32, meta_dim=16)
    
    bottlenecks = ['self_model_accuracy', 'core_reasoning']
    improvements = agent.meta_layer.generate_improvements(bottlenecks)
    
    assert len(improvements) > 0
    assert 'W_self' in improvements or 'W1' in improvements
    print(f"  Generated improvements for: {list(improvements.keys())}")
    print("✓ Improvement generation passed")


def test_save_load():
    """Test that save/load works"""
    print("\nTesting save/load...")
    agent = MetaCognitiveAgent(input_dim=32, hidden_dim=32, meta_dim=16)
    
    # Train briefly
    X_train, y_train = generate_synthetic_arc_tasks(n_tasks=10)
    agent.train(X_train, y_train, epochs=5, refinement_cycles=1)
    
    # Save
    agent.save_state("test_state.json")
    
    # Load into new agent
    new_agent = MetaCognitiveAgent(input_dim=32, hidden_dim=32, meta_dim=16)
    new_agent.load_state("test_state.json")
    
    # Verify weights match
    assert np.allclose(agent.meta_layer.W1, new_agent.meta_layer.W1)
    assert np.allclose(agent.meta_layer.W_out, new_agent.meta_layer.W_out)
    
    print("✓ Save/load passed")


def test_full_experiment():
    """Test the full experiment pipeline"""
    print("\nTesting full experiment pipeline...")
    
    # Generate data
    X_train, y_train = generate_synthetic_arc_tasks(n_tasks=100)
    X_test, y_test = generate_synthetic_arc_tasks(n_tasks=20)
    
    # Initialize and train
    agent = MetaCognitiveAgent(input_dim=32, hidden_dim=32, meta_dim=16)
    history = agent.train(X_train, y_train, epochs=20, refinement_cycles=2)
    
    # Evaluate
    self_awareness = agent.evaluate_self_awareness(X_test, y_test)
    meta_reasoning = agent.evaluate_meta_reasoning(X_test, y_test)
    
    print(f"  Self-Awareness: {self_awareness:.3f}")
    print(f"  Meta-Reasoning: {meta_reasoning:.3f}")
    
    assert 0 <= self_awareness <= 1
    assert 0 <= meta_reasoning <= 1
    print("✓ Full experiment passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("META-COGNITIVE AGENT TEST SUITE")
    print("=" * 60)
    
    try:
        test_agent_initialization()
        test_forward_pass()
        test_training()
        test_self_evaluation()
        test_bottleneck_identification()
        test_improvement_generation()
        test_save_load()
        test_full_experiment()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
