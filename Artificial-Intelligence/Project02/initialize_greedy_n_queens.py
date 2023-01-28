import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

def select_position(N, i, board, diagonals1, diagonals2):
    """
    This function selects a position for the queen in column i that minimizes conflicts.
    Ties are broken randomly.

    Inputs:
    N = board dimension
    i = index of the ith column where queen will be placed
    board = the board arrangement so far
    diagonals1, diagonals2 = two arrays that are used to help check diagonal conflicts

    Outputs:
    Returns a minimum conflict row position for the queen in column i.
    """

    # Variables to keep track of conflicts and minimum conflict value
    conflicts = np.zeros(N, dtype=int)
    min_conflict = float("inf")

    # Get conflicts for each row position that the queen can be in for that column
    for row_pos in range(N):
        # Count row conflicts
        row_conflicts = np.count_nonzero(board[:i] == row_pos)

        # Count diagonal conflicts
        diag_conflicts = np.count_nonzero(diagonals1[:i] == row_pos+i)
        diag_conflicts += np.count_nonzero(diagonals2[:i] == row_pos+N-i-1)

        # Get total number of conflicts
        conflict = row_conflicts + diag_conflicts
        conflicts[row_pos] = conflict
        
        # Update min conflict
        if conflict < min_conflict:
            min_conflict = conflict
    
    
    # Get indices with minimal conflicts
    min_conflict_positions = np.where(conflicts == min_conflict)
    # print(min_conflict_positions[0])

    # Break ties randomly
    return np.random.choice(min_conflict_positions[0])

    

def initialize_greedy_n_queens(N: int) -> list:
    """
    This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
    assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
    produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
    algorithm's performance!

    Example:
    Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

     _ _ _ _
    |Q|_|_|_|
    |_|_|Q|_|
    |_|_|_|Q|
    |_|Q|_|_|

    which has one diagonal conflict between its two rightmost columns.

    You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
    as any additional import statements will be removed by the autograder.

    :param N: integer representing the size of the NxN chessboard
    :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
    conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
    """
    greedy_init = np.zeros(N, dtype=int)
    # First queen goes in a random spot
    greedy_init[0] = np.random.randint(0, N)

    # Indices that are used to help determine diagonal conflicts
    indices = np.arange(0, N)
    indices_backward = np.arange(N-1, -1, -1)

    for i in range(1,N):
        # Initialize two sets of diagonal arrays to check for diagonal conflicts
        diagonals1 = np.add(greedy_init, indices)
        diagonals2 = np.add(greedy_init, indices_backward)

        # Select the next position for the queen greedily
        greedy_init[i] = select_position(N, i, greedy_init, diagonals1, diagonals2)

    return greedy_init


if __name__ == '__main__':
    # You can test your code here
    board = initialize_greedy_n_queens(10)
    print(board)