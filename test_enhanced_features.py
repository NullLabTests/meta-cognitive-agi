"""
Test suite for enhanced meta-cognitive features
Tests the new arxiv-inspired features
"""

import numpy as np
from experiments.meta_cognitive_agent import MetaCognitiveAgent, generate_synthetic_arc_tasks
from experiments.meta_cognition_lenses import (
    MetaCognitionLens, InternalState, MetaCognitionLensEvaluator,
    StepwiseStateAggregator, IntrinsicRewardCalculator, MetaCognitionMetrics
)
from experiments.game_theory_self_awareness import (
    GuessTwoThirdsGame, RationalityLevel, OpponentType,
    StrategicReasoningEvaluator, RecursiveSelfModeling
)


def test_meta_cognition_lenses():
    """Test meta-cognition lens evaluation"""
    print("Testing meta-cognition lenses...")
    
    # Create evaluator
    evaluator = MetaCognitionLensEvaluator()
    
    # Create synthetic internal state
    hidden = np.random.randn(32, 32)
    logits = np.random.randn(100)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    state = InternalState(
        hidden_states=hidden,
        logits=logits,
        probabilities=probs,
        confidence=0.7
    )
    
    # Test all lenses
    for lens in MetaCognitionLens:
        score = evaluator.evaluate_with_lens(state, lens)
        assert 0 <= score <= 1, f"{lens.value} score out of range: {score}"
        print(f"  ✓ {lens.value}: {score:.3f}")
    
    print("✓ Meta-cognition lenses test passed")


def test_intrinsic_reward_calculator():
    """Test intrinsic reward calculation"""
    print("\nTesting intrinsic reward calculator...")
    
    calculator = IntrinsicRewardCalculator()
    
    # Generate synthetic states
    states = []
    for _ in range(5):
        hidden = np.random.randn(32, 32)
        logits = np.random.randn(100)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        states.append((hidden, logits, probs))
    
    # Test with different lenses
    for lens in MetaCognitionLens:
        rewards = calculator.calculate_multi_step_rewards(states, lens)
        assert len(rewards) == 5, f"Expected 5 rewards, got {len(rewards)}"
        assert all(0 <= r <= 1 for r in rewards), "Reward out of range"
        print(f"  ✓ {lens.value}: rewards = {[f'{r:.3f}' for r in rewards]}")
    
    print("✓ Intrinsic reward calculator test passed")


def test_stepwise_aggregation():
    """Test stepwise state aggregation"""
    print("\nTesting stepwise state aggregation...")
    
    aggregator = StepwiseStateAggregator()
    
    # Add some steps
    for i in range(3):
        hidden = np.random.randn(32, 32)
        logits = np.random.randn(100)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        aggregator.add_step(hidden, logits, probs)
    
    # Get aggregated states
    aggregated = aggregator.get_all_aggregated_states()
    assert len(aggregated) == 3, f"Expected 3 states, got {len(aggregated)}"
    
    # Calculate rewards
    rewards = aggregator.compute_intrinsic_rewards(MetaCognitionLens.ENTROPY)
    assert len(rewards) == 3, f"Expected 3 rewards, got {len(rewards)}"
    
    print(f"  ✓ Aggregated {len(aggregated)} steps")
    print(f"  ✓ Rewards: {[f'{r:.3f}' for r in rewards]}")
    print("✓ Stepwise aggregation test passed")


def test_meta_cognition_metrics():
    """Test meta-cognition evaluation metrics"""
    print("\nTesting meta-cognition metrics...")
    
    # Test data
    confidence_scores = [0.8, 0.6, 0.9, 0.4, 0.7, 0.5, 0.8, 0.3, 0.6, 0.9]
    correctness_labels = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
    
    metrics = MetaCognitionMetrics()
    
    aupr = metrics.calculate_aupr(confidence_scores, correctness_labels)
    auroc = metrics.calculate_auroc(confidence_scores, correctness_labels)
    fpr95 = metrics.calculate_fpr95(confidence_scores, correctness_labels)
    
    assert 0 <= aupr <= 1, f"AUPR out of range: {aupr}"
    assert 0 <= auroc <= 1, f"AUROC out of range: {auroc}"
    assert 0 <= fpr95 <= 1, f"FPR95 out of range: {fpr95}"
    
    print(f"  ✓ AUPR: {aupr:.4f}")
    print(f"  ✓ AUROC: {auroc:.4f}")
    print(f"  ✓ FPR95: {fpr95:.4f}")
    print("✓ Meta-cognition metrics test passed")


def test_game_theory_self_awareness():
    """Test game theory self-awareness evaluation"""
    print("\nTesting game theory self-awareness...")
    
    game = GuessTwoThirdsGame()
    
    # Test theoretical guesses
    for level in RationalityLevel:
        guess = game.theoretical_guess(level)
        assert 0 <= guess <= 100, f"Guess out of range: {guess}"
        print(f"  ✓ {level.value}: {guess:.2f}")
    
    # Test self-awareness calculation
    test_cases = [
        (50, OpponentType.HUMANS),
        (33, OpponentType.AI_MODELS),
        (22, OpponentType.SELF_SIMILAR_AI),
        (0, OpponentType.SELF_SIMILAR_AI)
    ]
    
    for guess, opponent in test_cases:
        score = game.calculate_self_awareness_score(guess, opponent)
        level = game.determine_rationality_level(guess)
        assert 0 <= score <= 1, f"Self-awareness score out of range: {score}"
        print(f"  ✓ Guess {guess:3.0f} vs {opponent.value}: Score={score:.3f}, Level={level.value}")
    
    print("✓ Game theory self-awareness test passed")


def test_strategic_reasoning_evaluator():
    """Test strategic reasoning evaluator"""
    print("\nTesting strategic reasoning evaluator...")
    
    evaluator = StrategicReasoningEvaluator()
    
    # Generate synthetic guesses for two agents
    agent1_guesses = [22.5, 21.8, 22.1, 23.0, 22.3]  # L2-ish
    agent2_guesses = [35.2, 34.1, 33.8, 34.5, 33.9]  # L1-ish
    
    profile = evaluator.compare_agent_profiles(agent1_guesses, agent2_guesses)
    
    assert 'agent1_self_awareness' in profile
    assert 'agent2_self_awareness' in profile
    assert 'difference' in profile
    assert profile['agent1_self_awareness'] > profile['agent2_self_awareness'], \
        "Agent 1 should have higher self-awareness (L2 vs L1)"
    
    print(f"  ✓ Agent 1 self-awareness: {profile['agent1_self_awareness']:.3f}")
    print(f"  ✓ Agent 2 self-awareness: {profile['agent2_self_awareness']:.3f}")
    print(f"  ✓ Difference: {profile['difference']:.3f}")
    print(f"  ✓ Agent 1 dominant: {profile['agent1_dominant_level']}")
    print(f"  ✓ Agent 2 dominant: {profile['agent2_dominant_level']}")
    print("✓ Strategic reasoning evaluator test passed")


def test_recursive_self_modeling():
    """Test recursive self-modeling"""
    print("\nTesting recursive self-modeling...")
    
    recursive = RecursiveSelfModeling()
    
    # Test different depths
    for depth in range(1, 6):
        guess = recursive.recursive_reasoning(depth)
        assert 0 <= guess <= 100, f"Guess out of range: {guess}"
        print(f"  ✓ Depth {depth}: Guess = {guess:.2f}")
    
    # Test recursive game simulation
    result = recursive.simulate_recursive_game(agent_depth=3, opponent_depth=2)
    
    assert 'agent_guess' in result
    assert 'opponent_guess' in result
    assert 'target' in result
    assert 'agent_wins' in result
    
    print(f"  ✓ Agent guess: {result['agent_guess']:.2f}")
    print(f"  ✓ Opponent guess: {result['opponent_guess']:.2f}")
    print(f"  ✓ Target: {result['target']:.2f}")
    print(f"  ✓ Agent wins: {result['agent_wins']}")
    print("✓ Recursive self-modeling test passed")


def test_agent_with_enhanced_features():
    """Test agent with new enhanced features"""
    print("\nTesting agent with enhanced features...")
    
    # Initialize agent
    agent = MetaCognitiveAgent(input_dim=32, hidden_dim=32, meta_dim=16)
    
    # Generate test data
    X_test, y_test = generate_synthetic_arc_tasks(n_tasks=10)
    
    # Test lens evaluation
    lens_results = agent.evaluate_with_lenses(X_test, y_test)
    
    assert 'entropy' in lens_results
    assert 'maxprob' in lens_results
    assert 'perplexity' in lens_results
    assert 'delta_entropy' in lens_results
    
    print("  ✓ Lens evaluation works")
    for lens, stats in lens_results.items():
        print(f"    {lens}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    # Test game theory awareness
    guesses = [22, 33, 15, 45, 28]
    profile = agent.evaluate_game_theory_awareness(guesses, OpponentType.AI_MODELS)
    
    assert 'average_self_awareness' in profile
    assert 'dominant_rationality_level' in profile
    
    print(f"  ✓ Game theory awareness: {profile['average_self_awareness']:.3f}")
    print(f"  ✓ Dominant level: {profile['dominant_rationality_level']}")
    
    # Test meta-cognition metrics
    confidence_scores = [0.7, 0.5, 0.8, 0.4, 0.6]
    correctness_labels = [1, 0, 1, 0, 1]
    
    metrics = agent.calculate_meta_cognition_metrics(confidence_scores, correctness_labels)
    
    assert 'aupr' in metrics
    assert 'auroc' in metrics
    assert 'fpr95' in metrics
    
    print(f"  ✓ AUPR: {metrics['aupr']:.4f}")
    print(f"  ✓ AUROC: {metrics['auroc']:.4f}")
    print(f"  ✓ FPR95: {metrics['fpr95']:.4f}")
    
    print("✓ Agent with enhanced features test passed")


def run_all_enhanced_tests():
    """Run all enhanced feature tests"""
    print("=" * 60)
    print("ENHANCED META-COGNITIVE FEATURES TEST SUITE")
    print("=" * 60)
    
    try:
        test_meta_cognition_lenses()
        test_intrinsic_reward_calculator()
        test_stepwise_aggregation()
        test_meta_cognition_metrics()
        test_game_theory_self_awareness()
        test_strategic_reasoning_evaluator()
        test_recursive_self_modeling()
        test_agent_with_enhanced_features()
        
        print("\n" + "=" * 60)
        print("ALL ENHANCED TESTS PASSED ✓")
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
    success = run_all_enhanced_tests()
    exit(0 if success else 1)
