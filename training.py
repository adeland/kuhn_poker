import numpy as np
import logging
from tqdm import tqdm  # Progress bar library

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Constants for the Kuhn Poker game
PASS = 0
BET = 1
NUM_ACTIONS = 2  # Actions: PASS (0) or BET (1)
DECK = ['J', 'Q', 'K']  # The deck consists of three cards: Jack, Queen, King

class KuhnCFR:
    def __init__(self):
        self.regret_sum = {}  # Cumulative regrets for each info set
        self.strategy_sum = {}  # Sum of strategies over all iterations
        self.strategy = {}  # Current strategy for each info set
        self.average_strategy = {}  # Average strategy computed from strategy_sum

    def get_strategy(self, info_set):
        """Get the current strategy for an information set using regret matching."""
        if info_set not in self.regret_sum:
            self.regret_sum[info_set] = np.zeros(NUM_ACTIONS, dtype=np.float64)
            self.strategy_sum[info_set] = np.zeros(NUM_ACTIONS, dtype=np.float64)

        # Compute the strategy using regret matching
        regret = self.regret_sum[info_set]
        positive_regret = np.maximum(regret, 0)
        normalizing_sum = np.sum(positive_regret)

        if normalizing_sum > 0:
            strategy = positive_regret / normalizing_sum
        else:
            strategy = np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS  # Uniform distribution if no positive regret

        self.strategy[info_set] = strategy
        return strategy

    def update_strategy_sum(self, info_set, strategy, reach_probability):
        """Update the cumulative strategy sum for an information set."""
        if info_set not in self.strategy_sum:
            self.strategy_sum[info_set] = np.zeros(NUM_ACTIONS, dtype=np.float64)

        self.strategy_sum[info_set] += reach_probability * strategy

    def get_average_strategy(self):
        """Compute the average strategy from the cumulative strategy sum."""
        for info_set, strategy_sum in self.strategy_sum.items():
            normalizing_sum = np.sum(strategy_sum)
            if normalizing_sum > 0:
                self.average_strategy[info_set] = strategy_sum / normalizing_sum
            else:
                self.average_strategy[info_set] = np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS
        return self.average_strategy

    def cfr(self, history, player_card, opponent_card, reach_probabilities, player_to_act):
        """Recursive Counterfactual Regret Minimization algorithm."""
        plays = len(history)
        player = plays % 2  # Player 0 acts first, then alternates
        opponent = 1 - player

        # Check if the game has ended
        if plays >= 2:
            terminal_pass = history[-1] == PASS
            double_bet = history[-2:] == (BET, BET)
            is_player_card_higher = DECK.index(player_card) > DECK.index(opponent_card)

            if terminal_pass:
                if history == (PASS, PASS):
                    return 1 if is_player_card_higher else -1  # Payoff for showing down
                else:
                    return 1  # Last bettor wins the pot
            elif double_bet:
                return 2 if is_player_card_higher else -2  # Double bet payoff

        # Get the information set for the current player
        info_set = player_card + ''.join(map(str, history))

        # Get the current strategy for this information set
        strategy = self.get_strategy(info_set)
        action_utils = np.zeros(NUM_ACTIONS, dtype=np.float64)

        # Recursively call CFR for each possible action
        for a in range(NUM_ACTIONS):
            next_history = history + (a,)
            if player == player_to_act:
                action_utils[a] = -self.cfr(next_history, player_card, opponent_card,
                                            reach_probabilities, player_to_act)
            else:
                action_utils[a] = -self.cfr(next_history, opponent_card, player_card,
                                            reach_probabilities, player_to_act)

        # Update regret sums and strategy sums
        util = np.dot(strategy, action_utils)
        regrets = action_utils - util
        self.regret_sum[info_set] += reach_probabilities[player] * regrets
        self.update_strategy_sum(info_set, strategy, reach_probabilities[player])

        return util

    def train(self, iterations):
        """Train the bot using CFR for the specified number of iterations."""
        utilities = []
        with tqdm(total=iterations, desc="Training Progress", position=0) as pbar:  # Single-threaded progress bar
            for i in range(iterations):
                iteration_utility = 0
                for player_card in DECK:
                    for opponent_card in DECK:
                        if player_card != opponent_card:
                            # Initialize reach probabilities for both players
                            reach_probabilities = [1.0, 1.0]
                            utility = self.cfr((), player_card, opponent_card, reach_probabilities, 0)
                            iteration_utility += utility
                utilities.append(iteration_utility)

                # Log the average strategy every 5,000 iterations
                if (i + 1) % 5000 == 0:
                    self.get_average_strategy()
                    logging.info(f"Average Strategy at Iteration {i + 1}:")
                    for info_set, strategy in sorted(self.average_strategy.items()):
                        logging.info(f"{info_set}: {strategy}")

                # Update progress bar
                pbar.update(1)

        return utilities


# Train the bot
if __name__ == "__main__":
    kuhn_trainer = KuhnCFR()
    iterations = 500000  # Reduced iterations for faster execution
    utilities = kuhn_trainer.train(iterations)

    # Final output of the average strategy
    logging.info("Final Average Strategy:")
    for info_set, strategy in sorted(kuhn_trainer.average_strategy.items()):
        logging.info(f"{info_set}: {strategy}")
