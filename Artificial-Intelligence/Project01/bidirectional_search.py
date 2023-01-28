from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

class Frontier():
    """
    Class to help store the data and data structures for the frontier
    """
    def __init__(self, node):
        self.queue = deque()
        self.queue.append(node)
        self.set = {node.state}
        self.size = 1

def searchStep(problem, frontier1, explored1, frontier2, explored2, pathToAdvance, 
    path, num_nodes_expanded, max_frontier_size, pathCost):
    """
        Advance one step searching in the direction of 1 --> 2
        However, instead of using goal_test as with the BFS algorithm, we use a different stopping condition
    """
    
    node = frontier1.queue.popleft() # Get next node
    frontier1.size -= 1 # Decrement the size of the frontier
    frontier1.set.remove(node.state)
    explored1[node.state] = node # Add node to dictionary of explored nodes
    num_nodes_expanded += 1 # Increment the number of nodes expanded

    actions = problem.get_actions(node.state) # Get all actions
    
    for action in actions: # Loop through actions
        child = problem.get_child_node(node, action) # Get the child node
        actionCost = problem.action_cost(node.state, action, child.state) # Get action cost
        
        # Check if child has already been explored or is already in the frontier
        if ((child.state not in explored1) and (child.state not in frontier1.set)) or (node.path_cost + actionCost < child.path_cost): 
            
            child.path_cost = node.path_cost + actionCost # Update path cost            

            frontier1.queue.append(child) # Add child to frontier
            frontier1.size += 1 # Increment frontier size
            frontier1.set.add(child.state) # Add child to frontier set
            max_frontier_size = max(max_frontier_size, frontier1.size + frontier2.size) # Update the max frontier size
            
            # Check if the child is an intersection point and that the cost is less than the path cost 
            if (child.state in explored2) and ((child.path_cost + explored2[child.state].path_cost) < pathCost):

                goal = problem.goal_states[0]
                child2 = explored2[child.state]

                if pathToAdvance == 1: # If advancing from the source frontier
                    path = problem.trace_path(child) # Get the path from child to init node
                    path.extend(problem.trace_path(child2, goal)[-2::-1]) # Add in the path from child to goal state, 
                                                                        # making sure the child doesn't get included twice
                else: # Reverse if advancing from the destination frontier
                    path = problem.trace_path(child2)
                    path.extend(problem.trace_path(child, goal)[-2::-1])
                
                # Update path cost
                pathCost = child.path_cost + explored2[child.state].path_cost

    # Return    
    return path, num_nodes_expanded, max_frontier_size, pathCost


def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

        :param problem: instance of SimpleSearchProblem
        :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
                 num_nodes_expanded: number of nodes expanded by your search
                 max_frontier_size: maximum frontier size during search
        """

    # Initialize variables
    max_frontier_size = 0
    num_nodes_expanded = 0
    path = []

    # Create nodes based on initial and end state
    initNode = Node(parent=None, state=problem.init_state, action=None, path_cost=0)
    goalNode = Node(parent=None, state=problem.goal_states[0], action=None, path_cost=0)

    # Check if the goal is reached
    if initNode.state == goalNode.state:
        return [initNode.state], num_nodes_expanded, max_frontier_size

    # Initialize source frontier queue and explored set
    srcFrontier = Frontier(initNode)
    srcExplored = {}

    # Initialize destination frontier queue and explored set
    destFrontier = Frontier(goalNode)
    destExplored = {}
    
    # Initialize path cost and variable to store which frontier to advance next
    pathCost = float('inf')
    pathToAdvance = 1 # 1: advance forward direction next, -1: advance reverse direction next

    # Stopping condition is if the path costs of the top two nodes of the frontier queue are >= the path cost
    while srcFrontier.queue[0].path_cost + destFrontier.queue[0].path_cost < pathCost:

        # If both frontiers are empty, then return
        if srcFrontier.size == 0 and destFrontier.size == 0:
            if len(path) == 0:
                return None # No solution
            return path, num_nodes_expanded, max_frontier_size

        # Otherwise, advance the next step of one of the frontiers
        if pathToAdvance == 1: # Advance the src frontier
            path, num_nodes_expanded, max_frontier_size, pathCost = searchStep(problem, srcFrontier, srcExplored, 
                destFrontier, destExplored, pathToAdvance, path, num_nodes_expanded, max_frontier_size, pathCost)
        else: # Advance the dest frontier
            path, num_nodes_expanded, max_frontier_size, pathCost = searchStep(problem, destFrontier, destExplored, 
                srcFrontier, srcExplored, pathToAdvance, path, num_nodes_expanded, max_frontier_size, pathCost)
        pathToAdvance *= -1 # Alternate btwn src and dest frontiers
    
    # Return
    if len(path) == 0:
        return None # No solution
    return path, num_nodes_expanded, max_frontier_size

        


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
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('./stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 345
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Be sure to compare with breadth_first_search!