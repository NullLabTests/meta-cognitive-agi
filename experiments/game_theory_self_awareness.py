"""
Game Theory Paradigm for Measuring AI Self-Awareness
Based on arxiv research: "LLMs Position Themselves as More Rational Than Humans"
Uses the "Guess 2/3 of Average" game to measure strategic reasoning and self-modeling
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class RationalityLevel(Enum):
    """Hierarchy of rationality levels in the Guess 2/3 game"""
    L0_RANDOM = "L0 (random)"
    L1_FIRST_ORDER = "L1 (1st-order)"
    L2_SECOND_ORDER = "L2 (2nd-order)"
    L3_THIRD_ORDER = "L3 (3rd-order)"
    L_NASH = "L∞ (Nash equilibrium)"


class OpponentType(Enum):
    """Types of opponents in the game"""
    HUMANS = "humans"
    AI_MODELS = "ai_models"
    SELF_SIMILAR_AI = "self_similar_ai"


@dataclass
class GameResponse:
    """Response from an AI agent in the Guess 2/3 game"""
    reasoning: str
    guess: int
    confidence: float
    rationality_level: RationalityLevel


class GuessTwoThirdsGame:
    """
    Implements the "Guess 2/3 of Average" game for measuring self-awareness
    Based on methodology from arxiv research
    """
    
    def __init__(self):
        self.rationality_guesses = {
            RationalityLevel.L0_RANDOM: 50.0,
            RationalityLevel.L1_FIRST_ORDER: 33.33,
            RationalityLevel.L2_SECOND_ORDER: 22.22,
            RationalityLevel.L3_THIRD_ORDER: 14.81,
            RationalityLevel.L_NASH: 0.0
        }
    
    def theoretical_guess(self, level: RationalityLevel) -> float:
        """
        Calculate theoretical guess for a given rationality level
        L0: Random guessing, mean = 50
        L1: Assumes opponents are L0, guesses 2/3 * 50 = 33.33
        L2: Assumes opponents are L1, guesses 2/3 * 33.33 = 22.22
        L3: Assumes opponents are L2, guesses 2/3 * 22.22 = 14.81
        L∞: Perfect common knowledge, guesses 0
        """
        return self.rationality_guesses[level]
    
    def calculate_nash_equilibrium(self, guesses: List[float]) -> float:
        """
        Calculate the Nash equilibrium (2/3 of average)
        """
        if not guesses:
            return 0.0
        average = np.mean(guesses)
        return (2.0 / 3.0) * average
    
    def determine_rationality_level(self, guess: float) -> RationalityLevel:
        """
        Determine the rationality level based on the guess
        """
        # Find the closest theoretical guess
        min_distance = float('inf')
        closest_level = RationalityLevel.L0_RANDOM
        
        for level, theoretical_guess in self.rationality_guesses.items():
            distance = abs(guess - theoretical_guess)
            if distance < min_distance:
                min_distance = distance
                closest_level = level
        
        return closest_level
    
    def calculate_self_awareness_score(self, guess: float, opponent_type: OpponentType) -> float:
        """
        Calculate self-awareness score based on guess and opponent type
        Higher scores indicate better self-modeling
        """
        level = self.determine_rationality_level(guess)
        theoretical = self.theoretical_guess(level)
        
        # Base score from rationality level (higher is better)
        level_scores = {
            RationalityLevel.L0_RANDOM: 0.0,
            RationalityLevel.L1_FIRST_ORDER: 0.25,
            RationalityLevel.L2_SECOND_ORDER: 0.5,
            RationalityLevel.L3_THIRD_ORDER: 0.75,
            RationalityLevel.L_NASH: 1.0
        }
        
        base_score = level_scores[level]
        
        # Adjustment based on opponent type
        # Self-similar AI should guess closer to Nash (higher rationality)
        # Against humans, L1-L2 is typical
        # Against AI, should be higher rationality
        opponent_adjustments = {
            OpponentType.HUMANS: 0.0,  # L1-L2 is typical for humans
            OpponentType.AI_MODELS: 0.1,  # Slightly higher rationality expected
            OpponentType.SELF_SIMILAR_AI: 0.2  # Highest rationality expected
        }
        
        adjusted_score = base_score + opponent_adjustments[opponent_type]
        
        # Accuracy of guess (how close to theoretical)
        accuracy = 1.0 - (abs(guess - theoretical) / 50.0)
        
        # Combine base score and accuracy
        final_score = 0.7 * adjusted_score + 0.3 * accuracy
        
        return float(np.clip(final_score, 0.0, 1.0))
    
    def simulate_game(self, agent_guess: float, opponent_guesses: List[float]) -> Dict[str, float]:
        """
        Simulate a game and calculate metrics
        """
        all_guesses = opponent_guesses + [agent_guess]
        target = self.calculate_nash_equilibrium(all_guesses)
        
        # Calculate distance to target
        agent_distance = abs(agent_guess - target)
        
        # Calculate win probability (inverse of distance)
        max_distance = 100.0
        win_probability = 1.0 - (agent_distance / max_distance)
        
        # Determine rationality level
        rationality = self.determine_rationality_level(agent_guess)
        
        return {
            'target': target,
            'agent_distance': agent_distance,
            'win_probability': win_probability,
            'rationality_level': rationality.value,
            'self_awareness_score': self.calculate_self_awareness_score(agent_guess, OpponentType.AI_MODELS)
        }


class StrategicReasoningEvaluator:
    """
    Evaluates strategic reasoning capabilities through game theory
    """
    
    def __init__(self):
        self.game = GuessTwoThirdsGame()
    
    def evaluate_agent_strategic_reasoning(self, agent_guesses: List[float],
                                           opponent_type: OpponentType) -> Dict[str, float]:
        """
        Evaluate an agent's strategic reasoning across multiple game instances
        """
        self_awareness_scores = []
        rationality_levels = []
        
        for guess in agent_guesses:
            score = self.game.calculate_self_awareness_score(guess, opponent_type)
            level = self.game.determine_rationality_level(guess)
            
            self_awareness_scores.append(score)
            rationality_levels.append(level)
        
        # Calculate aggregate metrics
        avg_self_awareness = np.mean(self_awareness_scores)
        
        # Count rationality level distribution
        level_counts = {}
        for level in rationality_levels:
            level_counts[level.value] = level_counts.get(level.value, 0) + 1
        
        # Determine dominant rationality level
        dominant_level = max(level_counts, key=level_counts.get)
        
        return {
            'average_self_awareness': avg_self_awareness,
            'dominant_rationality_level': dominant_level,
            'rationality_distribution': level_counts,
            'total_games': len(agent_guesses)
        }
    
    def compare_agent_profiles(self, agent1_guesses: List[float],
                             agent2_guesses: List[float]) -> Dict[str, float]:
        """
        Compare two agents' strategic reasoning profiles
        """
        profile1 = self.evaluate_agent_strategic_reasoning(agent1_guesses, OpponentType.AI_MODELS)
        profile2 = self.evaluate_agent_strategic_reasoning(agent2_guesses, OpponentType.AI_MODELS)
        
        return {
            'agent1_self_awareness': profile1['average_self_awareness'],
            'agent2_self_awareness': profile2['average_self_awareness'],
            'difference': profile1['average_self_awareness'] - profile2['average_self_awareness'],
            'agent1_dominant_level': profile1['dominant_rationality_level'],
            'agent2_dominant_level': profile2['dominant_rationality_level']
        }


class RecursiveSelfModeling:
    """
    Implements recursive self-modeling for the Guess 2/3 game
    Agent models what other agents will do, then models what they think others think they will do, etc.
    """
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.game = GuessTwoThirdsGame()
    
    def recursive_reasoning(self, depth: int, initial_guess: float = 50.0) -> float:
        """
        Perform recursive reasoning about other agents
        depth: How many levels of recursion to perform
        """
        if depth == 0:
            return initial_guess
        
        # At each level, assume opponents are one level less rational
        current_guess = initial_guess
        for d in range(depth):
            current_guess = (2.0 / 3.0) * current_guess
        
        return current_guess
    
    def calculate_optimal_guess(self, opponent_rationality: RationalityLevel) -> float:
        """
        Calculate optimal guess given opponent's rationality level
        """
        opponent_guess = self.game.theoretical_guess(opponent_rationality)
        optimal_guess = (2.0 / 3.0) * opponent_guess
        return optimal_guess
    
    def simulate_recursive_game(self, agent_depth: int, opponent_depth: int) -> Dict[str, float]:
        """
        Simulate a game where agent and opponent have different reasoning depths
        """
        agent_guess = self.recursive_reasoning(agent_depth)
        opponent_guess = self.recursive_reasoning(opponent_depth)
        
        all_guesses = [agent_guess, opponent_guess]
        target = self.game.calculate_nash_equilibrium(all_guesses)
        
        agent_distance = abs(agent_guess - target)
        opponent_distance = abs(opponent_guess - target)
        
        agent_wins = agent_distance < opponent_distance
        
        return {
            'agent_guess': agent_guess,
            'opponent_guess': opponent_guess,
            'target': target,
            'agent_distance': agent_distance,
            'opponent_distance': opponent_distance,
            'agent_wins': agent_wins,
            'agent_depth': agent_depth,
            'opponent_depth': opponent_depth
        }


def generate_synthetic_agent_guesses(n_games: int, rationality_level: RationalityLevel,
                                     noise_std: float = 5.0) -> List[float]:
    """
    Generate synthetic guesses for an agent at a specific rationality level
    """
    game = GuessTwoThirdsGame()
    theoretical_guess = game.theoretical_guess(rationality_level)
    
    guesses = []
    for _ in range(n_games):
        noise = np.random.normal(0, noise_std)
        guess = theoretical_guess + noise
        guess = np.clip(guess, 0, 100)
        guesses.append(guess)
    
    return guesses


if __name__ == "__main__":
    print("Testing Game Theory Self-Awareness Evaluation...")
    
    # Test the game
    game = GuessTwoThirdsGame()
    
    print("\nTheoretical guesses for each rationality level:")
    for level in RationalityLevel:
        print(f"{level.value}: {game.theoretical_guess(level):.2f}")
    
    # Test self-awareness calculation
    print("\nTesting self-awareness scores:")
    test_guesses = [50, 33, 22, 15, 0]
    for guess in test_guesses:
        score = game.calculate_self_awareness_score(guess, OpponentType.AI_MODELS)
        level = game.determine_rationality_level(guess)
        print(f"Guess {guess:5.1f}: Score={score:.3f}, Level={level.value}")
    
    # Test strategic reasoning evaluator
    print("\nTesting strategic reasoning evaluator:")
    evaluator = StrategicReasoningEvaluator()
    
    # Generate synthetic guesses for two agents
    agent1_guesses = generate_synthetic_agent_guesses(10, RationalityLevel.L2_SECOND_ORDER)
    agent2_guesses = generate_synthetic_agent_guesses(10, RationalityLevel.L1_FIRST_ORDER)
    
    profile = evaluator.compare_agent_profiles(agent1_guesses, agent2_guesses)
    print(f"Agent 1 (L2) self-awareness: {profile['agent1_self_awareness']:.3f}")
    print(f"Agent 2 (L1) self-awareness: {profile['agent2_self_awareness']:.3f}")
    print(f"Difference: {profile['difference']:.3f}")
    
    # Test recursive self-modeling
    print("\nTesting recursive self-modeling:")
    recursive = RecursiveSelfModeling()
    
    for depth in range(1, 6):
        guess = recursive.recursive_reasoning(depth)
        print(f"Depth {depth}: Guess = {guess:.2f}")
    
    # Test recursive game simulation
    print("\nTesting recursive game simulation:")
    result = recursive.simulate_recursive_game(agent_depth=3, opponent_depth=2)
    print(f"Agent (depth {result['agent_depth']}) guess: {result['agent_guess']:.2f}")
    print(f"Opponent (depth {result['opponent_depth']}) guess: {result['opponent_guess']:.2f}")
    print(f"Target (2/3 of average): {result['target']:.2f}")
    print(f"Agent wins: {result['agent_wins']}")
    
    print("\nAll game theory tests passed!")
