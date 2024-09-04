MAX_FUSED_SIZE = 65536


def get_stride(array):
	strides = [1]
	for i in range(len(array.shape) - 1, 0, -1):  # Iterate in reverse shape order
		strides.insert(0, strides[0] * array.shape[i])
	return strides
