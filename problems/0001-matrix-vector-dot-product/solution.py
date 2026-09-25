def matrix_dot_vector(a, b):
    if len(a[0]) != len(b):
        return -1

    result = [0] * len(a)

    for i in range(len(a)):
        for k in range(len(b)):
            result[i] += a[i][k] * b[k]

    return result