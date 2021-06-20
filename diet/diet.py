# python3
from sys import stdin
from typing import ByteString


EPS = 1e-6
PRECISION = 0.001

def SolveEquation(a, b):
    n = len(b)
    x = [float(0) for i in range(n)]

    # Elimination
    # a is a list (row) of lists (column) of floats containing each coefficient of x_0 to x_i
    # b is a list of floats containing each constant b_0 to b_i
    for k in range(n-1):
        # the pivot value is the element at position k, k (i.e along the main diagonal on the k-th row)

        if abs(a[k][k]) < EPS:
            # The pivot value is too small a number (may cause division by 'zero' overflow errors)
            # iterate through the values below the pivot (same column, diff row) to find a value 
            # larger than the pivot to swap the rows
            for i in range(k+1, n):

                if abs(a[i][k]) > EPS:
                    # swap the i-th and k-th rows (don't forget the constants!)
                    a[i], a[k] = a[k], a[i]
                    b[i], b[k] = b[k], b[i]

        for i in range(k+1, n):
            # iterate through the values below the pivot to zero out i.e. eliminate the column
            # the value is already zero-ed and does not need to be multiplied by a factor
            if a[i][k] == 0: continue

            # find the factor to eliminate the element
            factor = a[k][k] / a[i][k]
            
            # multiply all values of the row with the factor and subtract from the corresponding
            # values of the pivot row in order to eliminate it
            for j in range(k, n):

                a[i][j] = a[k][j] - a[i][j]*factor
            
            # don't forget the constants!
            b[i] = b[k] - b[i]*factor

    #Back-substitution
    # All the values below the main diagonal have been eliminated
    # iteratively derive the solution by substituion
    try:
        x[n-1] = b[n-1] / a[n-1][n-1]
    except ZeroDivisionError:
        return []
    
    for i in range(n-2, -1, -1):
        
        sum_ax = 0

        for j in range(i+1, n):

            sum_ax += a[i][j]*x[j]
        
        try:
          x[i] = (b[i] - sum_ax) / a[i][i]
        except ZeroDivisionError:
          return []

    return x



def solve_diet_problem(n, m, A, b, c):  
  # n is the number of inequalties
  # m is the number of variables, which also gives an additional
  # m number of regular inequalities as each variable >= 0
  # A is a list of n lists of int containing the coefficients of each inequality 
  # b is a list of n int containing the constants of the inequality
  # i.e. the matrix A and vector b in Ax <= b
  # c is the unit pleasure value of each variable, the weighted sum of which is to be maximised

  # There are at most 8 inequalities (16 if you count in the regular inequalities)
  # with at most 8 variables. You can use this fact and the fact that the optimal 
  # solution is always in a vertex of the polyhedron corresponding to the linear 
  # programming problem. At least 𝑚 of the inequalities become equalities in each
  # vertex of the polyhedron.

  # The running time of this algorithm is 𝑂(2𝑛+𝑚(𝑚3+𝑚𝑛)), which is good enough to pass. 
  # 2𝑛+𝑚 is to go through all the subsets of the inequalities (although you will need only 
  # subsets of size 𝑚), 𝑚^3 is for Gaussian Elimination and 𝑚*𝑛 is to check asolution of 
  # a system of linear equations against all the inequalities. 
  from itertools import combinations
  from copy import deepcopy
  subsets = list(combinations(zip(A, b), m))

  
  solutions = []

  # take each possible subset of size 𝑚 out of all the 𝑛+𝑚 inequalities, 
  # and solve the system of linear equations where each 
  # equation is one of the selected inequalities changed to equality
  for system in subsets:
    # print(system)
    coeffs = []
    constants = []
    
    for equation in system:
      
      coeffs.append(deepcopy(equation[0]))
      constants.append(deepcopy(equation[1]))
    
    inf = max(constants) >= 10e9-1

    x = SolveEquation(coeffs, constants)
    
    valid = True

    # check whether this solution satisfies all the other inequalities
    if x:

      for item in x:
        if item < -PRECISION:
          valid = False
          break

      if valid:
        for i in range(n):
          check = sum([A[i][j] * x[j] for j in range(m)])
          
          # the solution does not satisfy this inequality
          if check > b[i]+PRECISION:
            valid = False
            break
      
        if valid:
          solutions.append([x, inf])

  if not solutions:
    return []

  # select the solution with the largest value of the total pleasure out of 
  # those which satisfy all inequalities
  import math
  max_pleasure = -math.inf
  best_solution = None

  for s in solutions:
    pleasure = sum([s[0][i] * c[i] for i in range(m)])

    if pleasure > max_pleasure:
      
      best_solution = s
      max_pleasure = pleasure

  # It is guaranteed that in all test cases in this problem, if a solution is bounded,
  # then amount1+amount2+···+amount𝑚 ≤ 10^9. Thus, to distinguish between bounded
  # and unbounded cases, just add this inequality to the initial problem.
  # If the resulting program has the last inequality among those 𝑚 that define the
  # vertex, output “Infinity”, otherwise output “Bounded solution” and the 
  # solution of the augmented problem on the second line.
  if best_solution[1]:
    return -1

  return best_solution

n, m = list(map(int, stdin.readline().split()))
A = []
for i in range(n):
  A += [list(map(float, stdin.readline().split()))]
b = list(map(float, stdin.readline().split()))
c = list(map(float, stdin.readline().split()))

# add the regular inequalities where each variable must be >= 0
for i in range(m):
  a = [float(0) if i != k else 1.0 for k in range(m)]
  A.append(a)
  b.append(float(0))
A.append([float(1) for k in range(m)])
b.append(10e9)

solution = solve_diet_problem(n, m, A, b, c)

if not solution:
  print("No solution")

else:
  if solution == -1:
    print("Infinity")
  else:
    print("Bounded solution")
    print(' '.join(map(str, solution[0])))




# 2 2
# 1 1 
# -1 -1
# 1 -2 
# 1 1

# 3 2
# -1 -1
# 1 0
# 0 1
# -1 2 2
# -1 2