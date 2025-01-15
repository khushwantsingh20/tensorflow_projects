import tensorflow as tf

# # Scalar (0-D tensor)
# scalar = tf.constant(42)
# print("Scalar1111111:", scalar)

# # Vector (1-D tensor)
# vector = tf.constant([1.0, 2.0, 3.0])
# print("Vector222222222:", vector)

# # Matrix (2-D tensor)
# matrix = tf.constant([[1, 2], [3, 4]])
# print("Matrix33333333:\n", matrix)

# # 3-D tensor
# tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# print("3-D Tensor4444444:\n", tensor_3d)




# # Addition
# a = tf.constant([1, 2, 3])
# b = tf.constant([4, 5, 6])
# print("Addition:", tf.add(a, b))

# # Multiplication
# print("Element-wise Multiplication:", tf.multiply(a, b))

# # Matrix multiplication
# matrix1 = tf.constant([[1, 2], [3, 4]])
# matrix2 = tf.constant([[5, 6], [7, 8]])
# print("Matrix Multiplication:\n", tf.matmul(matrix1, matrix2))


x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

# Calculate gradient of y with respect to x
grad = tape.gradient(y, x)
print("Gradient:", grad)
