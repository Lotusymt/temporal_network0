import numpy as np
import matplotlib.pyplot as plt
import math



class twoD_pre_estimator:
    def __init__(self, data):
        self.data = data
        # set size of potential values of a unit
        self.A = 100
        # store temporary data in the process of splitting
        self.tem_data = data

        self.rows = self.tem_rows = data.shape[0]
        self.cols = self.tem_cols = data.shape[1]
        self.D = self.rows * self.cols
        # store squares from the smallest to the biggest
        self.squares = []
        self.units = 0#originally 0, in case the matrix can be split into squares

        self.p_squares= []
        self.p_stages = []
        self.stage_predictability = []
        # split the rectangle matrix into square matrixes of maximum size through iteration

    def split_matrix(self):
        # quit iteration if the remainder is a square

        if self.tem_rows == self.tem_cols:
            self.squares.append(self.tem_data)
            return
        else:
            # if rows or cols =1, end iteration and record units
            if self.tem_rows == 1 or self.tem_cols == 1:
                self.units = self.tem_rows * self.tem_cols
                return
            # split the matrix into two parts

            if self.tem_rows > self.tem_cols:
                self.squares.append(self.tem_data[0:self.tem_cols, 0:self.tem_cols])
                self.tem_data = self.tem_data[self.tem_cols:self.tem_rows, 0:self.tem_cols]
                self.tem_rows = self.tem_rows - self.tem_cols
            else:
                self.squares.append(self.tem_data[0:self.tem_rows, 0:self.tem_rows])
                self.tem_data = self.tem_data[0:self.tem_rows, self.tem_rows:self.tem_cols]
                self.tem_cols = self.tem_cols - self.tem_rows
            # call the function again
            self.split_matrix()
    #calculate predictability of each square, attach them to p_squares
    def square_predictability(self):
        # calculate entropy of one square matrix
        # lambda_v is the smallest integer that makes no equal block found except in the exact same position
        def block_entropies(matrix):
            tem_rows, cols = matrix.shape
            n = tem_rows
            sum_lambda = 0
            for r in range(tem_rows):
                for c in range(cols):
                    v = r * cols + c
                    k = 1
                    # in the while loop. 1. check(r,c) can be expended to (r+k,c+k)
                    # 2. use a nested for loop to iterate the whole square matrix
                    # 3. if one of the inner loop finds a match, then break the while loop.
                    # otherwise, keep iterating and increase k by 1 when nested for loop finishes
                    while True:

                        lambda_v = k
                        # when start point reaches the up and right sides of the matrix, lambda_v=0
                        # so here I add a premise that when part of the indexes in v-C(k) is out of range, then the k is invalid
                        if not k < min(tem_rows - r, cols - c):
                            # now k==min(tem_rows-r,cols-c)
                            lambda_v = min(tem_rows - r, cols - c) - 1
                            break
                        block = matrix[r:r + k, c:c + k]

                        # hash might have collision, but it is very unlikely in quite limited topological space
                        block_hash = hash(block.tostring())
                        # mark if we should break the while loop
                        flag = True
                        for i in range(r):
                            for j in range(c):
                                # if the traversal block is out of range, then traverse the next block
                                if not k < min(tem_rows - r, cols - c):
                                    continue
                                if block_hash == hash(matrix[i:i + k, j:j + k].tostring()) and (i != r or j != c):
                                    # when one block is equal and not at the same place
                                    # then we can k++
                                    k += 1
                                    flag = False
                                    break
                                else:
                                    continue
                            break
                        if flag:
                            break

                    sum_lambda += lambda_v ** 2
            return (n ** 2 * np.log(n ** 2)) / sum_lambda
        def f(x,n,entropy_rate):
            return -x * math.log(x) - (1 - x) * math.log(1 - x) + (1 - x) * math.log(n - 1) - entropy_rate

        def f_prime(x,n):
            return -math.log(x) + math.log(1 - x) - math.log(n - 1)

        # calculate the predictability of one square matrix
        # newton method has a precision of 10^-6, no iteration limits
        def newton_raphson(x0,n,entropy_rate):
            x1 = x0 - f(x0,n,entropy_rate) / f_prime(x0,n)
            while abs(x1 - x0) > 1e-6:
                x0 = x1
                x1 = x0 - f(x0,n, entropy_rate) / f_prime(x0,n)
            return x1
        for i in range(len(self.squares)):
            x0 = 0.5  # initial guess
            # calculate the entropy rate of the square
            entropy_rate = block_entropies(self.squares[i])
            # calculate the predictability of the square
            predictability = newton_raphson(x0, self.squares[i].shape[0], entropy_rate)
            self.p_squares.append(predictability)

    #definition of stage is different from shown in Figure. S1. in supplementary material
    #stage zero is when the matrix is completely divided into squares; as stage++ one smallest square are merged into units
    def cal_stage_predictability(self, stage):
        p_i = 0
        # interate stage times from the biggest square
        for i in range(len(self.squares)-1,stage-1,-1):
            p_i += self.squares[i].shape[0] ** 2 * self.p_squares[i]
        #for squares splited to units and units
        tem_sum = 0
        for i in range(stage):
            tem_sum += self.squares[i].shape[0] ** 2
        tem_sum += self.units
        tem_sum /= self.D
        p_i = (p_i + tem_sum)/self.D
        return p_i

    def rec_stage_predictability(self):
        stage_num = len(self.squares)
        for i in range(stage_num):
            self.stage_predictability.append(self.cal_stage_predictability(i))

    #calculate the predictability of the whole matrix
    def cal_predictability(self):
        q = len(self.squares)
        #edges of squares
        #have to reverse it because recursion append the squares
        #from the smallest to the biggest, which is the reverse order of the paper
        e = [self.squares[i].shape[0] for i in range(q)].reverse()
        num_s_u = [q - i + sum(e[q - i + 1:]) + self.units for i in range(q)]
        k = ((q - 1) * sum([num_s_u[i] * self.stage_predictability[i] for i in range(q - 2)]) - sum(num_s_u[:q - 2]) * sum(self.stage_predictability[:q - 2])) / (
                    (q - 1) * sum([num_s_u[i] ** 2 for i in range(q - 2)]) - sum(num_s_u[:q - 2]) ** 2)
        b = (sum(self.stage_predictability[:q - 2]) - k * sum(num_s_u[:q - 2])) / (q - 1)
        result = k + b

        print(result)
        return result


