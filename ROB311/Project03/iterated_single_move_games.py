from abc import ABC, abstractmethod
import numpy as np
import time


class SingleMoveGamePlayer(ABC):
    """
    Abstract base class for a symmetric, zero-sum single move game player.
    """
    def __init__(self, game_matrix: np.ndarray):
        self.game_matrix = game_matrix
        self.n_moves = game_matrix.shape[0]
        super().__init__()

    @abstractmethod
    def make_move(self) -> int:
        pass


class IteratedGamePlayer(SingleMoveGamePlayer):
    """
    Abstract base class for a player of an iterated symmetric, zero-sum single move game.
    """
    def __init__(self, game_matrix: np.ndarray):
        super(IteratedGamePlayer, self).__init__(game_matrix)

    @abstractmethod
    def make_move(self) -> int:
        pass

    @abstractmethod
    def update_results(self, my_move, other_move):
        """
        This method is called after each round is played
        :param my_move: the move this agent played in the round that just finished
        :param other_move:
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class UniformPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(UniformPlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """

        :return:
        """
        return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        """
        The UniformPlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class FirstMovePlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(FirstMovePlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """
        Always chooses the first move
        :return:
        """
        return 0

    def update_results(self, my_move, other_move):
        """
        The FirstMovePlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class CopycatPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(CopycatPlayer, self).__init__(game_matrix)
        self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        """
        Always copies the last move played
        :return:
        """
        return self.last_move

    def update_results(self, my_move, other_move):
        """
        The CopyCat player simply remembers the opponent's last move.
        :param my_move:
        :param other_move:
        :return:
        """
        self.last_move = other_move

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        self.last_move = np.random.randint(self.n_moves)


def play_game(player1, player2, game_matrix: np.ndarray, N: int = 1000) -> (int, int):
    """

    :param player1: instance of an IteratedGamePlayer subclass for player 1
    :param player2: instance of an IteratedGamePlayer subclass for player 2
    :param game_matrix: square payoff matrix
    :param N: number of rounds of the game to be played
    :return: tuple containing player1's score and player2's score
    """
    p1_score = 0.0
    p2_score = 0.0
    n_moves = game_matrix.shape[0]
    legal_moves = set(range(n_moves))
    for idx in range(N):
        move1 = player1.make_move()
        move2 = player2.make_move()
        if move1 not in legal_moves:
            print("WARNING: Player1 made an illegal move: {:}".format(move1))
            if move2 not in legal_moves:
                print("WARNING: Player2 made an illegal move: {:}".format(move2))
            else:
                p2_score += np.max(game_matrix)
                p1_score -= np.max(game_matrix)
            continue
        elif move2 not in legal_moves:
            print("WARNING: Player2 made an illegal move: {:}".format(move2))
            p1_score += np.max(game_matrix)
            p2_score -= np.max(game_matrix)
            continue
        player1.update_results(move1, move2)
        player2.update_results(move2, move1)

        p1_score += game_matrix[move1, move2]
        p2_score += game_matrix[move2, move1]

    return p1_score, p2_score


class StudentAgent(IteratedGamePlayer):
    """
    This agent creates and updates an opponent's transition probability matrix that is used
    to estimate the opponent's strategy and select a move that will optimize the agent's results.
    
    To make a move, the agent uses the last move played by the opponent and the transition probability matrix
    to get an estimate of what the opponent's next move will be. For the first part of the game, the agent
    will just choose the move that counters the most probable next move of the opponent. Once in a while and 
    also near the end, the agent will randomly generate a move based on the probabilities of the opponent's 
    next moves.
    Note that the first two moves of the agent will be made randomly.

    After each play is made, the agent updates the opponent's transition probability matrix.

    To maximize the number of points against specific agents such as the copycat agent, the agent 
    tracks whether or not the opponent is a copycat agent and at a certain number of steps, it will
    change it's strategy to increase its reward.
    """
    def __init__(self, game_matrix: np.ndarray):
        """
        Initialize your game playing agent. here
        :param game_matrix: square payoff matrix for the game being played.
        """
        super(StudentAgent, self).__init__(game_matrix)
        
        # Parameters
        self.n_moves = game_matrix.shape[0] # Total number of possible moves
        self.table = np.zeros((self.n_moves, self.n_moves)) # Storing the opponents moves
        self.probabilities = np.zeros((self.n_moves, self.n_moves)) # Opponent's transition matrix
        self.opp_last_moves = [] # Opponent's last two moves [second last move, last move]
        self.my_last_move = None # My agent's last move
        self.step_number = 0 # Step number of the game
        self.response = {0:1, 1:2, 2:0} # Best response to opponent's move
        self.reward = 0 # Total reward

        # Check if opponent is copycat
        self.copycat_eval_step = 20 # Step number at which opponent is evaluated if its a copycat or not
        self.copycat_threshold = self.copycat_eval_step - 2 # Threshold to determine if opponent is a copycat
        self.opponent_is_copycat = False # Boolean to mark if opponent is a copycat
        self.opponent_copied_count = 0 # Counts the number of times opponent has copied my last move
        
    def make_move(self) -> int:
        """
        Play your move based on previous moves or whatever reasoning you want.
        :return: an int in (0, ..., n_moves-1) representing your move
        """
        move = None

        # Check if opponent is a copy-cat player and change the strategy if it is
        if self.step_number == self.copycat_eval_step:
            if self.opponent_copied_count > self.copycat_threshold:
                self.opponent_is_copycat = True
        if self.opponent_is_copycat:
            opp_move = self.my_last_move # opponent will copy my last move
            move = self.response[opp_move]

        if self.step_number > 1 and move is None:
            # Get the opponent's last move
            opp_last_move = self.opp_last_moves[-1]
            # Get the probability of the opponent's last move
            prob = self.probabilities[opp_last_move, :]

            # For earlier steps, we just choose the move with the highest probability
            if self.step_number < 800 and self.step_number % 100 != 0:
                # Get the most probable move
                opp_move = np.argmax(prob)
                # Get the corresponding move
                move = self.response[opp_move]
            
            # For later steps, we randomly choose a move based on their probabilities
            elif np.sum(prob) == 1:
                # Get the move with the highest probability
                opp_move = np.random.choice(np.arange(self.n_moves), p=prob)
                # Get the corresponding move
                move = self.response[opp_move]
        
        if move is None:
            # Get a random move
            move = np.random.randint(self.n_moves)

        self.step_number += 1

        return move

    def update_results(self, my_move, other_move):
        """
        Update your agent based on the round that was just played.
        :param my_move:
        :param other_move:
        :return: nothing
        """
        # Update the previous opponent steps
        self.opp_last_moves.append(other_move)
        if len(self.opp_last_moves) > 2:
            self.opp_last_moves.pop(0)

        # Update the opponent's transition matrix
        if len(self.opp_last_moves) == 2:
            self.table[self.opp_last_moves[0], self.opp_last_moves[1]] += 1
            self.probabilities = np.divide(self.table, np.maximum(1, self.table.sum(axis=1)).reshape(-1,1))

        # Update whether or not the opponent copied the last move
        if self.my_last_move == other_move:
            self.opponent_copied_count += 1

        # Update my last move
        self.my_last_move = my_move
        
        # Update reward
        self.reward += self.game_matrix[my_move, other_move]

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.).
        :return: nothing
        """
        # Parameters
        self.table = np.zeros((self.n_moves, self.n_moves))
        self.probabilities = np.zeros((self.n_moves, self.n_moves))
        self.opp_last_moves = []
        self.my_last_move = None
        self.step_number = 0
        self.reward = 0

        # Check if opponent is copycat
        self.opponent_is_copycat = False
        self.opponent_copied_count = 0

if __name__ == '__main__':
    """
    Simple test on standard rock-paper-scissors
    The game matrix's row (first index) is indexed by player 1 (P1)'s move (i.e., your move)
    The game matrix's column (second index) is indexed by player 2 (P2)'s move (i.e., the opponent's move)
    Thus, for example, game_matrix[0, 1] represents the score for P1 when P1 plays rock and P2 plays paper: -1.0 
    because rock loses to paper.
    """
    game_matrix = np.array([[0.0, -1.0, 1.0],
                            [1.0, 0.0, -1.0],
                            [-1.0, 1.0, 0.0]])
    uniform_player = UniformPlayer(game_matrix)
    first_move_player = FirstMovePlayer(game_matrix)
    uniform_score, first_move_score = play_game(uniform_player, first_move_player, game_matrix)

    print("Uniform player's score: {:}".format(uniform_score))
    print("First-move player's score: {:}".format(first_move_score))

    # Now try your agent
    student_player = StudentAgent(game_matrix)
    student_scores = []
    first_move_scores = []
    start = time.time()
    for i in range(100):
        student_score, first_move_score = play_game(student_player, first_move_player, game_matrix)
        student_scores.append(student_score)
        first_move_scores.append(first_move_score)
    end = time.time()
    print("Your player's score: {:}".format(sum(student_scores)))
    print("First-move player's score: {:}".format(sum(first_move_scores)))
    print("Time (s):", end-start)


    # Now try your agent
    student_player.reset()
    student_scores = []
    uniform_scores = []
    start = time.time()
    for i in range(100):
        student_score, uniform_score = play_game(student_player, uniform_player, game_matrix)
        student_scores.append(student_score)
        uniform_scores.append(uniform_score)
    end = time.time()
    print("Your player's score: {:}".format(sum(student_scores)))
    print("Uniform player's score: {:}".format(sum(uniform_scores)))
    print("Time (s):", end-start)