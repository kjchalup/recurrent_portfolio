import numpy
from kerpy.GaussianKernel import GaussianKernel
from independence_testing.HSICSpectralTestObject import HSICSpectralTestObject

# def compute_independence(x, y):
#     n_samples = x.shape[0]
#     kernel_x = GaussianKernel() 
#     kernel_y = GaussianKernel()
#     block_size = min(n_samples, 1000)
#     test_object = HSICBlockTestObject(n_samples, kernelX=kernel_x, 
#                                  kernelY=kernel_y, 
#                                  kernelX_use_median=True,
#                                  kernelY_use_median=True,
#                                  blocksize=block_size,
#                                  nullvarmethod='permutation')
#     return test_object.compute_pvalue(x, y)

def compute_independence(x, y):
    n_samples = x.shape[0]
    kernel_x = GaussianKernel() 
    kernel_y = GaussianKernel()
    test_object = HSICSpectralTestObject(n_samples, kernelX=kernel_x, 
                                 kernelY=kernel_y, 
                                 kernelX_use_median=True,
                                 kernelY_use_median=True,
                                 rff=True, num_rfx=20, num_rfy=20,
                                 num_nullsims=1000)
    return test_object.compute_pvalue(x, y)
