def test_g(student_func):
    
    try:
        search = student_func
        
    except:
        #TODO put in a more relevant message
        return 2 #You didn't define a function called search
    
    try:
        grid = [[0, 1],
                [0, 0]]
        init = [0, 0]
        goal = [len(grid)-1,len(grid[0])-1]
        cost = 1
        
        user_answer = search(grid,init,goal,cost)
        if not user_answer:
            return 3 # didn't return anything
    except:
        return 103 # Problem.
    
    try:
        grid = [[0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 0]]
        init = [0, 0]
        goal = [len(grid)-1,len(grid[0])-1]
        cost = 1
    
        user_answer = search(grid,init,goal,cost)
        if user_answer != [11, 4, 5]:
            return 4 # Your code didn't work for example in lecture
    except:
        return 104
    
    try:
        grid = [[0, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0]]
        init = [0, 0]
        goal = [len(grid)-1,len(grid[0])-1]
        cost = 1
        user_answer = search(grid,init,goal,cost)
        if user_answer != [17, 4, 5]:
            return 5
    except:
        return 105
    
    try:
        grid = [[0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0]]
        init = [0, 0]
        goal = [len(grid)-1,len(grid[0])-1]
        cost = 1
        user_answer = search(grid,init,goal,cost)
        if type(user_answer) != str or user_answer.lower() != 'fail':
            return 6 # Your code didn't return 'fail' when it should have.
    except:
        return 106
    
    return 0 # correct


def run_grader(student_func):
    
    grade_result = dict()
    
    # test grids
    
    grid_1 =[[0, 1],
             [0, 0]]
    
    grid_2 = [[0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0]]
    
    grid_3 = [[0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0]]
   
    grid_4 = [[0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]]

    try:
        result = test_g(student_func)

        comment = ""
        if result == 0:
            comment = ""
        elif result == 1:
            comment = "There was an error running your solution. Please make sure there are no syntax errors, \nindentation errors, etc. and try again."
        elif result == 2:
            comment = "search is not defined"
        elif result == 2.5:
            comment = "search did not return a list of three integers or the string 'fail'"
        elif result % 100 == 3:
            if result == 3:
                comment = f"search did not return anything for grid: {grid_1}"
            else:
                comment = f"search raised an exception for grid: {grid_1}"
        elif result % 100 == 4:
            if result == 4:
                comment = f"search didn't return the expected output for grid: {grid_2}"
            else:
                comment = f"search raised an exception for grid: {grid_2}"
       
        elif result % 100 == 5:
            if result == 5:
                comment = f"search didn't return the expected output for grid: {grid_3}"
            else:
                comment = f"search raised an exception for grid: {grid_3}"
      
        elif result % 100 == 6:
            if result == 6:
                comment = f"search didn't return the expected output for grid: {grid_4}"
            else:
                comment = f"search raised an exception for grid: {grid_4}"
            
        grade_result['correct'] = (result == 0)
        if grade_result['correct']:
            grade_result['comment'] = "Correct! " + comment
        else:
            grade_result['comment'] = comment
    except:
        grade_result['correct'] = False
        grade_result['comment'] = """There was an error running your solution. Make sure that 
    search takes four arguments: grid, init, goal, cost. Also
    make sure that you are not using any global variables other
    than delta and delta_name."""
    
    return grade_result.get('comment')
