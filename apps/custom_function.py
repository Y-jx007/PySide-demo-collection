from custom_import import *

def vectorize_func(func):
            """将mp函数向量化"""
            def vectorized_func(z_array,*args, **kwargs):
                if np.isscalar(z_array):
                    return complex(func(z_array,*args, **kwargs))
                
                result = np.empty(z_array.shape, dtype=complex)
                flat_z = z_array.flat
                flat_result = result.flat
                
                for i in range(len(flat_z)):
                    flat_result[i] = complex(func(flat_z[i],*args, **kwargs))
                
                return result
            return vectorized_func