import cupy as cp

if __name__ == '__main__':

    # Define Gaussian kernel parameters (adjust sigma as needed)
    kernel_size = 5
    sigma = 0.5

    # Create Gaussian kernel on CPU (example using NumPy)
    import numpy as np

    gaussian_kernel = np.fromfunction(lambda x, y: np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2)),
                                      (kernel_size, kernel_size))
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize for correct weighting

    # Convert kernel to cuPy array
    gaussian_kernel_gpu = cp.array(gaussian_kernel)



    # Apply convolution with the Gaussian kernel on GPU
    blurred_image = cp.nn.conv2d(image_gpu, gaussian_kernel_gpu, pad_mode='same')
