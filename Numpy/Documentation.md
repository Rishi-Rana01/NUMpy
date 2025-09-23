# NumPy Basics: Complete Step-by-Step Documentation

## Introduction
This documentation provides a comprehensive, beginner-friendly guide to NumPy fundamentals based on the content in `Numpy/phase-1.ipynb`. NumPy (Numerical Python) is a powerful library for numerical computing in Python, offering efficient array operations, mathematical functions, and tools for data manipulation. This guide covers array creation, performance comparisons, data structures, properties, and reshaping techniques with clear explanations and examples.

Whether you're new to NumPy or looking to refresh your knowledge, this step-by-step breakdown will help you understand each concept with practical code snippets and expected outputs.

## 1. NumPy Array Basics
NumPy arrays are the core data structure in the library, providing a fast and flexible way to work with numerical data. Unlike Python lists, NumPy arrays are homogeneous (all elements must be of the same type) and support vectorized operations for better performance.

### Step 1: Importing NumPy
To use NumPy, you need to import it. The standard alias is `np` for convenience.

```python
import numpy as np
```

### Step 2: Creating Arrays from Lists
You can create NumPy arrays directly from Python lists. This is useful for converting existing data into NumPy format.

#### Creating a 1D Array
```python
arr_1d = np.array([1, 2, 3, 4, 5])
print("1D array: ", arr_1d)
```
**Explanation**: This creates a one-dimensional array from a list of integers. The output will be: `1D array: [1 2 3 4 5]`.

#### Creating a 2D Array
```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D array: \n", arr_2d)
```
**Explanation**: This creates a two-dimensional array (matrix) from a list of lists. The output will be:
```
2D array:
 [[1 2 3]
  [4 5 6]]
```

## 2. List vs NumPy Array
Python lists and NumPy arrays behave differently, especially in operations like multiplication. NumPy arrays support element-wise operations, making them more efficient for numerical tasks.

### Step 1: Multiplication Comparison
```python
py_list = [1, 2, 3, 4, 5]
print("Python List multiplication: ", py_list * 2)

np_array = np.array([1, 2, 3, 4, 5])
print("Numpy Array multiplication: ", np_array * 2)
```
**Explanation**:
- Python list multiplication repeats the list: `Python List multiplication: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]`.
- NumPy array multiplication performs element-wise multiplication: `Numpy Array multiplication: [2 4 6 8 10]`.

### Step 2: Performance Comparison
NumPy is optimized for large-scale operations. Let's compare the time taken for a simple operation on a large dataset.

```python
import time

# Python list
start = time.time()
py_list = [i * 2 for i in range(1000000)]
print("Python list time: ", time.time() - start)

# NumPy array
start = time.time()
np_array = np.arange(1000000) * 2
print("Numpy array time: ", time.time() - start)
```
**Explanation**: This measures the time for multiplying each element by 2. NumPy arrays are significantly faster due to vectorized operations and optimized C code. Expected output shows NumPy taking much less time (e.g., NumPy array time: 0.001 seconds vs. Python list time: 0.1 seconds).

## 3. Creating Arrays from Scratch
NumPy provides several functions to create arrays with specific values or patterns, without needing to start from a list.

### Step 1: Array of Zeros
```python
zeros = np.zeros((3, 4))
print("Array of Zeros: \n", zeros)
```
**Explanation**: Creates a 3x4 array filled with zeros. Output:
```
Array of Zeros:
 [[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]
```

### Step 2: Array of Ones
```python
ones = np.ones((3, 4))
print("Array of Ones: \n", ones)
```
**Explanation**: Creates a 3x4 array filled with ones. Output:
```
Array of Ones:
 [[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]
```

### Step 3: Array Filled with a Specific Value
```python
full = np.full((3, 4), 7)
print("Array of Full: \n", full)
```
**Explanation**: Creates a 3x4 array where every element is 7. Output:
```
Array of Full:
 [[7 7 7 7]
  [7 7 7 7]
  [7 7 7 7]]
```

### Step 4: Identity Matrix
```python
eye = np.eye(4)
print("Identity Matrix: \n", eye)
```
**Explanation**: Creates a 4x4 identity matrix (diagonal elements are 1, others are 0). Output:
```
Identity Matrix:
 [[1. 0. 0. 0.]
  [0. 1. 0. 0.]
  [0. 0. 1. 0.]
  [0. 0. 0. 1.]]
```

### Step 5: Random Array
```python
random = np.random.random((3, 4))
print("Array of Randoms: \n", random)
```
**Explanation**: Creates a 3x4 array with random values between 0 and 1. Output varies each time.

### Step 6: Sequence Array
```python
sequence = np.arange(10, 21, 2)
print("Sequence Array: \n", sequence)
```
**Explanation**: Creates an array with values from 10 to 20 (exclusive) with a step of 2. Output: `Sequence Array: [10 12 14 16 18 20]`.

## 4. Vector, Matrix, Tensor
NumPy supports multi-dimensional arrays, which are essential for representing different types of data structures.

### Step 1: Vector (1D Array)
```python
vector = np.array([1, 2, 3])
print("Vector: ", vector)
```
**Explanation**: A vector is a 1D array. Output: `Vector: [1 2 3]`.

### Step 2: Matrix (2D Array)
```python
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Matrix: \n", matrix)
```
**Explanation**: A matrix is a 2D array. Output:
```
Matrix:
 [[1 2 3]
  [4 5 6]
  [7 8 9]]
```

### Step 3: Tensor (3D Array)
```python
tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("Tensor: \n", tensor)
```
**Explanation**: A tensor is a 3D array, useful for representing data like images or volumes. Output:
```
Tensor:
 [[[1 2]
   [3 4]]

  [[5 6]
   [7 8]]]
```

## 5. Array Properties
Understanding array properties helps you inspect and manipulate data effectively.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Shape: ", arr.shape)      # Number of rows and columns
print("Dimension: ", arr.ndim)   # Number of dimensions
print("Size: ", arr.size)        # Total number of elements
print("Data Type: ", arr.dtype)  # Data type of elements
```
**Explanation**:
- **Shape**: `(2, 3)` – 2 rows, 3 columns.
- **Dimension**: `2` – 2D array.
- **Size**: `6` – Total elements.
- **Data Type**: `int64` (or similar, depending on system).

## 6. Array Reshaping
Reshaping allows you to change the structure of an array without altering its data.

### Step 1: Basic Reshaping
```python
arr = np.arange(12)
print("Original array: ", arr)

reshaped = arr.reshape((3, 4))
print("Reshaped array: \n", reshaped)
```
**Explanation**: `np.arange(12)` creates `[0 1 2 ... 11]`. Reshaping to (3,4) gives:
```
Reshaped array:
 [[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]
```

### Step 2: Flattening
```python
flattened = reshaped.flatten()
print("Flattened array: ", flattened)
```
**Explanation**: Converts the 2D array back to 1D. Output: `Flattened array: [ 0  1  2  3  4  5  6  7  8  9 10 11]`.

### Step 3: Raveling
```python
raveled = reshaped.ravel()
print("Raveled array: ", raveled)
```
**Explanation**: Similar to flatten, but returns a view (not a copy), which is more memory-efficient. Output: `Raveled array: [ 0  1  2  3  4  5  6  7  8  9 10 11]`.

### Step 4: Transposing
```python
transpose = reshaped.T
print("Transposed array: \n", transpose)
```
**Explanation**: Swaps rows and columns. Output:
```
Transposed array:
 [[ 0  4  8]
  [ 1  5  9]
  [ 2  6 10]
  [ 3  7 11]]
```

## Conclusion
This documentation covers the essential NumPy concepts from `Numpy/phase-1.ipynb`, including array creation, performance benefits, data structures, properties, and reshaping. By following these step-by-step examples, you can build a strong foundation in NumPy for data science, machine learning, and scientific computing.

For further learning, explore NumPy's official documentation or experiment with these examples in a Jupyter notebook. If you have questions or need advanced topics, refer to additional resources!

**Key Takeaways**:
- NumPy arrays are faster and more efficient than Python lists for numerical operations.
- Use functions like `np.zeros`, `np.ones`, and `np.arange` for quick array creation.
- Properties like `shape` and `ndim` help you understand your data.
- Reshaping operations like `reshape` and `transpose` are crucial for data manipulation.
