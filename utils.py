from operator import mul


def transpose(matrix):
    return list(map(list, zip(*matrix)))


def make_mat(val, height, width):
    return [[val for _ in range(width)] for _ in range(height)]


def mat_mul(mat1, mat2):
    result = make_mat(0, len(mat1), len(mat2[0]))

    for rowIdx, row in enumerate(mat1):
        for colIdx, c in enumerate(mat2[0]):
            col = list(map(lambda r: r[colIdx], mat2))
            result[rowIdx][colIdx] = sum(list(map(mul, row, col)))

    return result
