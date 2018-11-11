import copy

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

def adjacent_cells(grid,row,col):
    yMax = len(grid)-1
    xMax = len(grid[0])-1
    
    ret = []
    
    if row-1 >= 0 and grid[row-1][col] != 1:
        ret.append((row-1,col))
    if row+1 <= yMax and grid[row+1][col] != 1:
        ret.append((row+1,col))
    if col-1 >= 0 and grid[row][col-1] != 1:
        ret.append((row,col-1))
    if col+1 <= xMax and grid[row][col+1] != 1:
        ret.append((row,col+1))
        
    return ret

def compute_value(grid,init):
    yMax = len(grid)-1
    xMax = len(grid[0])-1
    
    indices = [ (row,col) for row in range(yMax+1) for col in range(xMax+1) ]
    G = { index : {'d':-1} for index in indices }

    first_cell = (init[0],init[1])
    
    G[first_cell]['d'] = 0
    to_check = { first_cell : G[first_cell] }
    current_cell = first_cell

    while len(to_check) > 0:
        G[current_cell] = to_check.pop(current_cell)
        for cell in adjacent_cells(grid,*current_cell):
            if G[cell]['d'] != -1: # means cell has already been checked
                continue
            if cell in to_check: # don't add the cell again
                continue
            to_check[cell] = G[cell]
            G[cell]['d'] = G[current_cell]['d'] + 1
        if len(to_check) > 0:
            current_cell = min( to_check.keys(), key=lambda k: to_check[k]['d'] )

    return [ [ G[(row,col)]['d'] for col in range(xMax+1) ] for row in range (yMax+1) ]

def is_valid_answer(grid,init,user_answer):
    # check for correct length
    if len(grid) != len(user_answer):
        return False
    for i in range(len(grid)):
        if len(grid[i]) != len(user_answer[i]):
            return False
    height = len(grid)
    width = len(grid[0])
    
    # unreachable cells have value -1
    value_grid = compute_value(grid,init)
    
    # check that unreachable cells are marked with -1
    reachable_cells = 0
    for i in range(height):
        for j in range(width):
            if value_grid[i][j] == -1 and user_answer[i][j] != -1:
                return False
            elif value_grid[i][j] >= 0:
                reachable_cells += 1
    
    # check that every number from 0 to reachable_cells-1 is in user_answer
    present = [0]*reachable_cells
    for i in range(height):
        for j in range(width):
            if user_answer[i][j] < 0:
                continue
            else:
                present[user_answer[i][j]] = 1
    if sum(present) != reachable_cells:
        return False
    
    # check that the numbers occur in a legal pattern
    # (the expansion number of a cell should be at least the number of steps
    # away from init it takes to get to the cell)
    for i in range(height):
        for j in range(width):
            if user_answer[i][j] < 0:
                continue
            elif user_answer[i][j] < value_grid[i][j]:
                return False
    
    return True

def tests(student_func):
    
    try:
        search = student_func
    except:
        return 2 #You didn't define a function called search
    
    try:
        grid = [[0, 1],
                [0, 0]]
        init = [0,0]
        goal = [len(grid)-1,len(grid[0])-1]
        cost = 1
        
        user_answer = search(grid,init,goal,cost)
        if not user_answer:
            return 3 # Your function didn't return anything.
    except:
        return 103 # problem
    
    try:
        grid = [[0, 1, 1, 1, 1],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 1, 1, 0],
                [0, 0, 0, 1, 0]]
        init = [0,0]
        goal = [len(grid)-1,len(grid[0])-1]
        cost = 1
        
        G = copy.deepcopy(grid)
        
        user_answer = search(G,init,goal,cost)
        if not is_valid_answer(grid,init,user_answer):
            return 4 # Your code didn't work for example in lecture
    except:
        return 104
    
    try:
        grid = [[0, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 0, 0]]
        init = [0,0]
        goal = [len(grid)-1,len(grid[0])-1]
        cost = 1
        G = copy.deepcopy(grid)
        
        user_answer = search(G,init,goal,cost)
        if not is_valid_answer(grid,init,user_answer):
            return 5
    except:
        return 105

    try:
        grid = [[0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1, 0]]
        init = [0, 0]
        goal = [len(grid)-1, len(grid[0])-1]
        cost = 1
        G = copy.deepcopy(grid)
        
        user_answer = search(G,init,goal,cost)
        if not is_valid_answer(grid,init,user_answer):
            return 6
    except:
        return 106
    
    return 0
