from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of SimpleSearchProblem
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

    # Check if the goal is reached
    if problem.goal_test(node.state):
        return [node.state], num_nodes_expanded, max_frontier_size

    # Initialize frontier queue and update frontier size
    frontier = deque()
    frontier.append(node)
    frontierSize = 1
    max_frontier_size = 1

    # Initialize explored set
    explored = set()

    while True:
        if len(frontier) == 0:
            return None # No solution found

        node = frontier.popleft() # Get next node
        explored.add(node.state) # Add node to the set of explored nodes
        num_nodes_expanded += 1 # Increment the number of nodes expanded
        frontierSize -= 1 # Update the frontier size


        actions = problem.get_actions(node.state) # Get all actions
        
        for action in actions: # Loop through actions
            child = problem.get_child_node(node, action) # Get the child node
            
            # Check if child has already been explored or is already in the frontier
            if (child.state not in explored) and (child not in frontier): 
                
                if problem.goal_test(child.state): 
                    # Return solution if goal is reached
                    path = problem.trace_path(child)
                    return path, num_nodes_expanded, max_frontier_size

                frontier.append(child) # Add child to frontier
                frontierSize += 1 # Update the frontier size
                max_frontier_size = max(max_frontier_size, frontierSize) # Update the max frontier size
    


if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('./stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)