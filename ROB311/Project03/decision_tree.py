import numpy as np
# DO NOT ADD TO OR MODIFY ANY IMPORT STATEMENTS


def dt_entropy(goal, examples):
    """
    Compute entropy over discrete random varialbe for decision trees.
    Utility function to compute the entropy (wich is always over the 'decision'
    variable, which is the last column in the examples).

    :param goal: Decision variable (e.g., WillWait), cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the entropy of the decision variable, given examples.
    """
    # INSERT YOUR CODE HERE.
    entropy = 0.
    # Be careful to check the number of examples
    # Avoid NaN examples by treating the log2(0.0) = 0

    N = len(examples)

    # Calculate entropy
    #
    # Note that for the binary case: 
    # q = p / (p + n)
    # B(q) = -(q*log2(q) + (1-q)*log2(1-q))
    # entropy = B(q)
    for i in range(len(goal[1])):
        q = np.count_nonzero(examples[:,-1] == i) / N
        if q != 0:
            entropy -= q * np.log2(q)

    return entropy


def dt_cond_entropy(attribute, col_idx, goal, examples):
    """
    Compute the conditional entropy for attribute. Utility function to compute the conditional entropy (which is always
    over the 'decision' variable or goal), given a specified attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the conditional entropy, given the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    cond_entropy = 0.0
    N = len(examples)

    # Calculate the conditional entropy using the formula:
    # H(V|X) = - sum of p(xi, vj)*log2(p(xi, vj) / p(xi))
    for i in range(len(attribute[1])):
        # Determine p(xi)
        pi = np.count_nonzero(examples[:,col_idx] == i) / N
        
        for j in range(len(goal[1])):
            # Determine p(xi, vj)
            pij = np.count_nonzero(np.logical_and(examples[:,col_idx] == i, examples[:,-1] == j)) / N
            #print("i:", i, "j:", j, "pi:", pi, "pij:", pij)

            # Update the conditional entropy
            if pij == 0 or pi == 0:
                continue
            else:
                cond_entropy -= pij * np.log2(pij/pi)

    return cond_entropy


def dt_info_gain(attribute, col_idx, goal, examples):
    """
    Compute information gain for attribute.
    Utility function to compute the information gain after splitting on attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the information gain, given the attribute and examples.

    """
    # INSERT YOUR CODE HERE.
    info_gain = 0.

    # Calculate info gain as entropy - remainder
    entropy = dt_entropy(goal, examples)
    remainder = dt_cond_entropy(attribute, col_idx, goal, examples)
    info_gain = entropy - remainder

    return info_gain


def dt_intrinsic_info(attribute, col_idx, examples):
    """
    Compute the intrinsic information for attribute.
    Utility function to compute the intrinsic information of a specified attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the intrinsic information for the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    # Be careful to check the number of examples
    # Avoid NaN examples by treating the log2(0.0) = 0
    intrinsic_info = 0.

    # Determine Intrinsic Info using the formula:
    # IntrinsicInfo(A) = - sum of [(pk + nk)/(p + n)] * log2((pk + nk)/(p + n))
    N = len(examples) # Total number of examples N = p + n
    for k in range(len(attribute[1])):
        # Determine Nk = pk + nk
        Nk = np.count_nonzero(examples[:,col_idx] == k)

        # Update intrinsic info
        qk = Nk / N
        if qk != 0:
            intrinsic_info -= qk * np.log2(qk)

    return intrinsic_info


def dt_gain_ratio(attribute, col_idx, goal, examples):
    """
    Compute information gain ratio for attribute.
    Utility function to compute the gain ratio after splitting on attribute. Note that this is just the information
    gain divided by the intrinsic information.
    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the gain ratio, given the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    # Avoid NaN examples by treating 0.0/0.0 = 0.0
    gain_ratio = 0.

    gain = dt_info_gain(attribute, col_idx, goal, examples)
    intrinsic_info = dt_intrinsic_info(attribute, col_idx, examples)

    if gain == 0 or intrinsic_info == 0:
        return gain_ratio
    
    gain_ratio = gain / intrinsic_info

    return gain_ratio


def learn_decision_tree(parent, attributes, goal, examples, score_fun):
    """
    Recursively learn a decision tree from training data.
    Learn a decision tree from training data, using the specified scoring function to determine which attribute to split
    on at each step. This is an implementation of the algorithm on pg. 702 of AIMA.

    :param parent: Parent node in tree (or None if first call of this algorithm).
    :param attributes: Attributes avaialble for splitting at this node.
    :param goal: Goal, decision variable (classes/labels).
    :param examples: Subset of examples that reach this point in the tree.
    :param score_fun: Scoring function used (dt_info_gain or dt_gain_ratio)
    :return: Root node of tree structure.
    """
    # YOUR CODE GOES HERE
    node = None
    # 1. Do any examples reach this point?
    if len(examples) <= 0:
        label = plurality_value(goal, parent.examples)
        node = TreeNode(parent, None, None, True, label)

    # 2. Or do all examples have the same class/label? If so, we're done!
    elif np.all(examples[:,-1] == examples[0,-1]):
        label = examples[0,-1]
        node = TreeNode(parent, None, None, True, label)

    # 3. No attributes left? Choose the majority class/label.
    elif not any(attributes):
        label = plurality_value(goal, examples)
        node = TreeNode(parent, None, None, True, label)

    # 4. Otherwise, need to choose an attribute to split on, but which one? Use score_fun and loop over attributes!
    else:
        # Best score?
        best_score = - float('inf')
        best_index = 0
        for i, attribute in enumerate(attributes):
            if attribute:
                score = score_fun(attribute, i, goal, examples)
                if score > best_score:
                    best_score = score
                    best_index = i
        #print("scores", scores, best_score, best_index)

        # NOTE: to pass the Autolab tests, when breaking ties you should always select the attribute with the smallest (i.e.
        # leftmost) column index!
        best_attribute = attributes[best_index]
        #print(best_attribute)

        # Create a new internal node using the best attribute, something like:
        # node = TreeNode(parent, attributes[best_index], examples, False, 0)
        node = TreeNode(parent, best_attribute, examples, False, 0)

        # Now, recurse down each branch (operating on a subset of examples below).
        # You should append to node.branches in this recursion
        for value in range(len(best_attribute[1])):
            examples_subset = examples[examples[:,best_index] == value]
            attributes_subset = [False if attribute == best_attribute else attribute for attribute in attributes]
            subtree = learn_decision_tree(node, attributes_subset, goal, examples_subset, score_fun)
            node.branches.append(subtree)
    
    return node


def plurality_value(goal: tuple, examples: np.ndarray) -> int:
    """
    Utility function to pick class/label from mode of examples (see AIMA pg. 702).
    :param goal: Tuple representing the goal
    :param examples: (n, m) array of examples, each row is an example.
    :return: index of label representing the mode of example labels.
    """
    vals = np.zeros(len(goal[1]))

    # Get counts of number of examples in each possible attribute class first.
    for i in range(len(goal[1])):
        vals[i] = sum(examples[:, -1] == i)

    return np.argmax(vals)


class TreeNode:
    """
    Class representing a node in a decision tree.
    When parent == None, this is the root of a decision tree.
    """
    def __init__(self, parent, attribute, examples, is_leaf, label):
        # Parent node in the tree
        self.parent = parent
        # Attribute that this node splits on
        self.attribute = attribute
        # Examples used in training
        self.examples = examples
        # Boolean representing whether this is a leaf in the decision tree
        self.is_leaf = is_leaf
        # Label of this node (important for leaf nodes that determine classification output)
        self.label = label
        # List of nodes
        self.branches = []

    def query(self, attributes: np.ndarray, goal, query: np.ndarray) -> (int, str):
        """
        Query the decision tree that self is the root of at test time.

        :param attributes: Attributes available for splitting at this node
        :param goal: Goal, decision variable (classes/labels).
        :param query: A test query which is a (n,) array of attribute values, same format as examples but with the final
                      class label).
        :return: label_val, label_txt: integer and string representing the label index and label name.
        """
        node = self
        while not node.is_leaf:
            b = node.get_branch(attributes, query)
            node = node.branches[b]

        return node.label, goal[1][node.label]

    def get_branch(self, attributes: list, query: np.ndarray):
        """
        Find attributes in a set of attributes and determine which branch to use (return index of that branch)

        :param attributes: list of attributes
        :param query: A test query which is a (n,) array of attribute values.
        :return:
        """
        for i in range(len(attributes)):
            if self.attribute[0] == attributes[i][0]:
                return query[i]
        # Return None if that attribute can't be found
        return None

    def count_tree_nodes(self, root=True) -> int:
        """
        Count the number of decision nodes in a decision tree.
        :param root: boolean indicating if this is the root of a decision tree (needed for recursion base case)
        :return: number of nodes in the tree
        """
        num = 0
        for branch in self.branches:
            num += branch.count_tree_nodes(root=False) + 1
        return num + root

def print_tree(root):
    """
    Function used for testing purposes to print the tree
    """
    if root:
        if root.is_leaf:
            print("Decision:", root.label)
        else:
            print("Attribute:", root.attribute)
            for node in root.branches:
                print_tree(node)

if __name__ == '__main__':
    # Example use of a decision tree from AIMA's restaurant problem on page (pg. 698)
    # Each attribute is a tuple of 2 elements: the 1st is the attribute name (a string), the 2nd is a tuple of options
    a0 = ('Alternate', ('No', 'Yes'))
    a1 = ('Bar', ('No', 'Yes'))
    a2 = ('Fri-Sat', ('No', 'Yes'))
    a3 = ('Hungry', ('No', 'Yes'))
    a4 = ('Patrons', ('None', 'Some', 'Full'))
    a5 = ('Price', ('$', '$$', '$$$'))
    a6 = ('Raining', ('No', 'Yes'))
    a7 = ('Reservation', ('No', 'Yes'))
    a8 = ('Type', ('French', 'Italian', 'Thai', 'Burger'))
    a9 = ('WaitEstimate', ('0-10', '10-30', '30-60', '>60'))
    attributes = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
    # The goal is a tuple of 2 elements: the 1st is the decision's name, the 2nd is a tuple of options
    goal = ('WillWait', ('No', 'Yes'))

    # Let's input the training data (12 examples in Figure 18.3, AIMA pg. 700)
    # Each row is an example we will use for training: 10 features/attributes and 1 outcome (the last element)
    # The first 10 columns are the attributes with 0-indexed indices representing the value of the attribute
    # For example, the leftmost column represents the attribute 'Alternate': 0 is 'No', 1 is 'Yes'
    # Another example: the 3rd last column is 'Type': 0 is 'French', 1 is 'Italian', 2 is 'Thai', 3 is 'Burger'
    # The 11th and final column is the label corresponding to the index of the goal 'WillWait': 0 is 'No', 1 is 'Yes'
    examples = np.array([[1, 0, 0, 1, 1, 2, 0, 1, 0, 0, 1],
                         [1, 0, 0, 1, 2, 0, 0, 0, 2, 2, 0],
                         [0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 1],
                         [1, 0, 1, 1, 2, 0, 1, 0, 2, 1, 1],
                         [1, 0, 1, 0, 2, 2, 0, 1, 0, 3, 0],
                         [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
                         [0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0],
                         [0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 1],
                         [0, 1, 1, 0, 2, 0, 1, 0, 3, 3, 0],
                         [1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                         [1, 1, 1, 1, 2, 0, 0, 0, 3, 2, 1]])

    # print("entropy", dt_entropy(goal, examples))
    # print("cond_entropy", dt_cond_entropy(a1, 1, goal, examples))
    # print("info gain", dt_info_gain(a1, 1, goal, examples))
    # print("intrinsic_info", dt_intrinsic_info(a1, 1, examples))
    # print("gain_ratio", dt_gain_ratio(a1, 1, goal, examples))

    # Build your decision tree using dt_info_gain as the score function
    tree = learn_decision_tree(None, attributes, goal, examples, dt_info_gain)
    print_tree(tree)
    # Query the tree with an unseen test example: it should be classified as 'Yes'
    test_query = np.array([0, 0, 1, 1, 2, 0, 0, 0, 2, 3])
    _, test_class = tree.query(attributes, goal, test_query)
    print("Result of query: {:}".format(test_class))

    # Repeat with dt_gain_ratio:
    tree_gain_ratio = learn_decision_tree(None, attributes, goal, examples, dt_gain_ratio)
    print_tree(tree_gain_ratio)
    # Query this new tree: it should also be classified as 'Yes'
    _, test_class = tree_gain_ratio.query(attributes, goal, test_query)
    print("Result of query with gain ratio as score: {:}".format(test_class))
