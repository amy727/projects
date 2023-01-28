from collections import deque
from support import definite_clause

### THIS IS THE TEMPLATE FILE
### WARNING: DO NOT CHANGE THE NAME OF FILE OR THE FUNCTION SIGNATURE


def pl_fc_entails(symbols_list : list, KB_clauses : list, known_symbols : list, query : int) -> bool:
    """
    pl_fc_entails function executes the Propositional Logic forward chaining algorithm (AIMA pg 258).
    It verifies whether the Knowledge Base (KB) entails the query
        Inputs
        ---------
            symbols_list  - a list of symbol(s) (have to be integers) used for this inference problem
            KB_clauses    - a list of definite_clause(s) composed using the numbers present in symbols_list
            known_symbols - a list of symbol(s) from the symbols_list that are known to be true in the KB (facts)
            query         - a single symbol that needs to be inferred

            Note: Definitely check out the test below. It will clarify a lot of your questions.

        Outputs
        ---------
        return - boolean value indicating whether KB entails the query
    """

    ### START: Your code
    # Initialize variables
    count = []
    inferred = {}
    agenda = deque()

    # Set the number of symbols in the symbol's premise in the count table
    for clause in KB_clauses:
        count.append([clause, len(clause.body)])

    # Initialize false for all symbols in inferred table
    for symbol in symbols_list:
        inferred[symbol] = False

    # Add known symbols to the agenda queue
    for knownSymbol in known_symbols:
        agenda.append(knownSymbol)

    # Loop through and check the inferences
    while agenda:
        p = agenda.popleft() # pop from the agenda queue

        if p == query: # if the wanted query, return true
            return True

        if inferred[p] == False: # if the inferred[p] if false
            inferred[p] = True

            for i in range(len(KB_clauses)): # for clause in KB
                clause = KB_clauses[i] 

                if p in clause.body: # if p is in c.PREMISE
                    count[i][1] -= 1 # decrement count[c]
                    if count[i][1] == 0:
                        agenda.append(clause.conclusion)

    return False
    ### END: Your code


# SAMPLE TEST
if __name__ == '__main__':

    # Symbols used in this inference problem (Has to be Integers)
    symbols = [1,2,9,4,5]

    # Clause a: 1 and 2 => 9
    # Clause b: 9 and 4 => 5
    # Clause c: 1 => 4
    KB = [definite_clause([1, 2], 9), definite_clause([9,4], 5), definite_clause([1], 4)]

    # Known Symbols 1, 2
    known_symbols = [1, 2]

    # Does KB entail 5?
    entails = pl_fc_entails(symbols, KB, known_symbols, 5)

    print("Sample Test: " + ("Passed" if entails == True else "Failed"))
