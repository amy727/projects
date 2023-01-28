import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem

def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    
    # Initialize variables
    max_frontier_size = 0
    num_nodes_expanded = 0
    path = []

    # Create first node based on initial state
    node = Node(parent=None, state=problem.init_state, action=None, path_cost=0)

    # Initialize frontier priority queue
    frontier = queue.PriorityQueue()
    nodeHeuristic = problem.heuristic(problem.init_state) # Get the node heuristic
    frontier.put((node.path_cost + nodeHeuristic, node))
    frontierSize = 1 # Initialize the frontier size

    # Initialize a frontier dictionary for easy searching
    frontierSet = {node.state: node.path_cost + nodeHeuristic} 

    # Initialize the explored set
    explored = set()

    while True:
        if frontierSize == 0:
            return None # No solution found
        
        # Get next node from priority queue
        node = frontier.get()[1] 
        frontierSize -= 1

        # Return solution if goal is reached
        if problem.goal_test(node.state): 
            path = problem.trace_path(node)
            return path, num_nodes_expanded, max_frontier_size
        
        explored.add(node.state) # Add node to the set of explored nodes
        
        actions = problem.get_actions(node.state) # Get all actions
        num_nodes_expanded += 1 # Increment the number of nodes expanded
        
        for action in actions: # Loop through actions
            
            child = problem.get_child_node(node, action) # Get the child node
            heuristic = problem.heuristic(child.state) # Get the heuristic
            
            # Update the frontier if the child hasn't been explored and isn't in frontier already
            # or if the one in the frontier has a higher cost
            if ( ((child.state not in explored) and (child.state not in frontierSet)) or
                ((child.state in frontierSet) and (frontierSet[child.state] > child.path_cost + heuristic)) ): 

                frontier.put((child.path_cost + heuristic, child)) # Add child to frontier
                frontierSize += 1 # Increment the frontier size
                frontierSet[child.state] = child.path_cost + heuristic # Update the frontier set
                max_frontier_size = max(max_frontier_size, frontierSize) # Update the max frontier size



def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    transition_start_probability = 0.35
    transition_end_probability = 0.45
    peak_nodes_expanded_probability = 0.40
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.1
    M = 700
    N = 700
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)

    # Experiment and compare with BFS