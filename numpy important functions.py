import numpy as np
# rowvector = np.array([10, 20, 30])
# columnvector = np.array([[10], [20], [30]])
# print(rowvector)
# print(columnvector)

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Matrix", matrix)
# select 3rd row and 3rd column
print("Matrix", matrix[2, 2])
# select first 2 rows and all columns
print("Matrix", matrix[:2, :])
# view the number of rows and columns
print("Rows and columns", matrix.shape)
# view the number of elements row*columns
print("Total elements", matrix.size)
# view the no. of dimension 2 in this case
print("Dimension", matrix.ndim)
# return the maximum element
# -------------------------------Max and Min--------------------------------------
print("max elemnt", np.max(matrix))
# return the min element
print("min elemnt", np.min(matrix))
# return the maximum element in each columns
print("max elemnt in each column", np.max(matrix, axis=0))
# return the maximum element in each row
print("max elemnt in each row", np.max(matrix, axis=1))

# -------------------------------Calculate Average(Mean)--------------------------------------
print("Average (mean) ", np.mean(matrix))

# -------------------------------Reshape--------------------------------------

print("9,1 reshape", matrix.reshape(9, 1))
# here -1 says as many columns as needed and 1 row
print("1,-1 reshape", matrix.reshape(1, -1))
# if we provide only 1 value reshpae would return a 1D array of that length
print("with only 1 argument i.e 9", matrix.reshape(9))
# we can also use flatten method to convert a matrix in 1d array
print(matrix.flatten())


# -------------------------------Transpose--------------------------------------
print("Transpose of matrix:", matrix.T)
# -------------------------------Diagonal--------------------------------------
print("DIAGONAL of matrix:", matrix.diagonal())
# -------------------------------Dot product--------------------------------------
vector_1 = [1, 2, 3]
vector_2 = [7, 8, 9]
print("dot product is", np.dot(vector_1, vector_2))

# -------------------------------adding, subtrating an multiplying--------------------------------------

matrix_2 = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])

print("add 2 matrices", np.add(matrix, matrix_2))
print("subtract 2 matrices", np.add(matrix, matrix_2))
# Multiplication elemnt wise not dot product
print(matrix * matrix_2)
# multiplication row column wise
print(" multiplication row column wise", np.matmul(matrix, matrix_2))

# -------------------------------matrix with all zeroes and one--------------------------------------
# gives all zeros matrix of 4 by 4
zeros = np.zeros([4, 4])
# gives all ones matrix of 4 by 4
zeros = np.ones([4, 4])

# -------------------------------Generate Ranom values--------------------------------------
#  generate 3 random integers between 1 and 10
print(np.random.randint(1, 11, 3))

# draw 6 numbers from a normal distribution with mean 1.0 an std 2.0
print(np.random.normal(1.0, 2.0, 6))
