# python3

EPS = 1e-6

def ReadEquation():
    size = int(input())
    a = []
    b = []
    for row in range(size):
        line = list(map(float, input().split()))
        a.append(line[:size])
        b.append(line[size])
    return a, b

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

                if abs(a[i][k]) > abs(a[k][k]):
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
    x[n-1] = b[n-1] / a[n-1][n-1]
    
    for i in range(n-2, -1, -1):
        
        sum_ax = 0

        for j in range(i+1, n):

            sum_ax += a[i][j]*x[j]
        
        x[i] = (b[i] - sum_ax) / a[i][i]

    return x



if __name__ == "__main__":
    a, b = ReadEquation()
    
    x = SolveEquation(a, b)
    
    print(' '.join(map(str, x)))

    exit(0)

