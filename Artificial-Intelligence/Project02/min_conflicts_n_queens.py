import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

def find_conflicts(N, board, diagonals1, diagonals2):
    """
    This function finds all the queens that has conflicts and returns a random conflicting queen.
    If there are no conflicts, the function returns None.

    Inputs:
    N = board dimension
    i = index of the ith column where queen will be placed
    board = the board arrangement so far
    diagonals1, diagonals2 = two arrays that are used to help check diagonal conflicts

    Outputs:
    Returns None if there are no conflicts.
    Otherwise, returns a conflicting column index.
    """

    # Set of column indices that conflict
    conflicts = set()

    # Loop through the board and get any conflicts
    for i in range(N):
        
        # Get any row conflicts associated with the queen in col i
        row_conflicts = np.count_nonzero(board == board[i]) - 1
        if row_conflicts > 0:
            conflicts.add(i) # Add conflicting indices to set

        # Get any diagonal conflicts associated with the queen in col i
        diag_conflicts = np.count_nonzero(diagonals1 == board[i]+i) - 1
        diag_conflicts += np.count_nonzero(diagonals2 == board[i]+N-i-1) - 1
        if diag_conflicts > 0:
            conflicts.add(i) # Add conflicting indices to set

    # Break ties randomly
    if len(conflicts) > 0:
        return np.random.choice(list(conflicts))
    return None

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
        row_conflicts = np.count_nonzero(board == row_pos) - 1

        # Count diagonal conflicts
        diag_conflicts = np.count_nonzero(diagonals1 == row_pos+i) - 1
        diag_conflicts += np.count_nonzero(diagonals2 == row_pos+N-i-1) - 1

        # Get total number of conflicts
        conflict = row_conflicts + diag_conflicts
        conflicts[row_pos] = conflict
        
        # Update min conflict
        if conflict < min_conflict:
            min_conflict = conflict
    
    
    # Get indices with minimal conflicts
    min_conflict_positions = np.where(conflicts == min_conflict)

    # Break ties randomly
    return np.random.choice(min_conflict_positions[0])

def min_conflicts_n_queens(initialization: list) -> (list, int):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """
    # Initialize variables
    N = len(initialization)
    solution = initialization.copy()
    num_steps = 0
    max_steps = 1000

    # Indices that are used to help determine diagonal conflicts
    indices = np.arange(0, N)
    indices_backward = np.arange(N-1, -1, -1)

    for idx in range(max_steps):
        # Initialize two sets of diagonal arrays to check for diagonal conflicts
        diagonals1 = np.add(solution, indices)
        diagonals2 = np.add(solution, indices_backward)

        # Select a random conflicting queen
        var = find_conflicts(N, solution, diagonals1, diagonals2)
        
        # If no conflicts, return solution
        if var == None:
            return (solution, num_steps)

        # Otherwise, select a minimal conflicting position for the queen and increment step count
        solution[var] = select_position(N, var, solution, diagonals1, diagonals2)
        #plot_n_queens_solution(solution)
        num_steps += 1

    return ([], -1)


if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    N = 10
    # Use this after implementing initialize_greedy_n_queens.py
    assignment_initial = initialize_greedy_n_queens(N)
    # Plot the initial greedy assignment
    #plot_n_queens_solution(assignment_initial)
    print(assignment_initial)

    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)
    # Plot the solution produced by your algorithm
    #plot_n_queens_solution(assignment_solved)
    print(assignment_solved, n_steps)
