import cupy as np


def get_median_filter(images):
    images = np.asarray(images)

    # Reshape to combine frames along a new axis (efficient for stacking)
    result = images.transpose([1, 2, 0]).astype(np.uint8)

    # Reshape the array to flatten each element pair
    flattened_arr = result.reshape(-1, len(images))
    # Calculate the median along the columns (axis=1)
    medians = np.median(flattened_arr, axis=1)
    # Reshape the medians back to the original
    median_array = medians.reshape(result.shape[0], result.shape[1])
    return median_array.astype(np.uint8)


if __name__ == '__main__':
    images = np.array([
        [[1, 2, 3],
         [4, 10, 6],
         [7, 8, 9]],

        [[1, 2, 3],
         [4, 20, 6],
         [7, 80, 9]],

        [[1, 2, 3],
         [4, 40, 64],
         [7, 20, 9]],

        [[1, 2, 30],
         [4, 22, 64],
         [7, 20, 9]]
    ])

    result = get_median_filter(images)
    print(result)
    print(result.dtype)
