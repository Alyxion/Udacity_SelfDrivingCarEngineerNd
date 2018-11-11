import test

def run_grader_grid(student_func):
    
    grade_result = dict()

    try:
        result = test.tests(student_func)
        correct = result == 0
        comment = ""    
        if result == 0:
            comment = ""
        elif result == 1:
            comment = "There was an error running your solution. Please make sure there are no syntax errors, \nindentation errors, etc. and try again."
        elif result == 2:
            comment = "search is not defined"
        elif result == 3:
            comment = "search did not return anything"
        elif result % 100 == 4:
            if result == 4:
                comment = "search didn't return the expected output for:\ngrid = ["
            else:
                comment = "search raised an exception for:\ngrid = ["
            grid = [[0, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 0, 1, 0]]
            for i in range(len(grid)):
                comment += str(grid[i])
                if i < len(grid)-1:
                    comment += ',\n        '
                else:
                    comment += ']'
        elif result % 100 == 5:
            if result == 5:
                comment = "search didn't return the expected output for:\ngrid = ["
            else:
                comment = "search raised an exception for:\ngrid = ["
            grid = [[0, 1, 0, 0, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 1],
                    [0, 0, 0, 1, 0, 0, 0]]
            for i in range(len(grid)):
                comment += str(grid[i])
                if i < len(grid)-1:
                    comment += ',\n        '
                else:
                    comment += ']'
        elif result % 100 == 6:
            if result == 6:
                comment = "search didn't return the expected output for:\ngrid = ["
            else:
                comment = "search raised an exception for:\ngrid = ["
            grid = [[0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0]]
            for i in range(len(grid)):
                comment += str(grid[i])
                if i < len(grid)-1:
                    comment += ',\n        '
                else:
                    comment += ']'
   
        grade_result['correct'] = correct
        if correct:
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