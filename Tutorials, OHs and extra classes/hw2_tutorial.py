import numpy as np

def einsum():
    matrix = np.arange(12).reshape(3,4)
    sum = np.sum(matrix, axis=1)
    print('Matrix: \n', matrix, '\n\n', 'Sum: ', sum)

    v1 = np.arange(4)
    v2 = np.arange(4) + 2

    result = np.einsum('w,w->w', v1, v2)
    print(f'{v1} X {v2} = {result}')

    #### Output
    #[0 1 2 3] X [2 3 4 5] = [ 0  3  8 15]

    result = np.einsum('w,w', v1, v2)
    print(f'{v1} X and + {v2} = {result}')

    #### Output
    #[0 1 2 3] X and + [2 3 4 5] = 26
    # a = np.array([0,1,2])
    # np.einsum('i,ij->i', matrix, matrix)

def stride():
    input = np.arange(25).reshape(5,5)
    print(input, f'\nStrides: {input.strides}\n')

    # Output: 
    # [[ 0  1  2  3  4]
    #  [ 5  6  7  8  9]
    #  [10 11 12 13 14]
    #  [15 16 17 18 19]
    #  [20 21 22 23 24]] 
    # Strides: (40, 8)

    kernel_size = 3
    layer_stride = 2

    height, width = input.shape
    rows_stride, columns_strides = input.strides

    out_height = int((height-kernel_size)/layer_stride + 1)
    out_width = int((width-kernel_size)/layer_stride + 1)

    new_shape = (out_height, out_width, kernel_size, kernel_size)
    new_strides = (rows_stride * layer_stride, columns_strides * layer_stride, rows_stride, columns_strides)

    windowed_input = np.lib.stride_tricks.as_strided(input, new_shape, new_strides)
    print(windowed_input, f'\nShape: {windowed_input.shape}, \tStrides: {windowed_input.strides}')

    
def main():
    stride()
    #einsum()

if __name__=="__main__":
    main()