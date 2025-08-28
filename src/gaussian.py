import numpy as np
import math
import matplotlib.pyplot as plt

g = make_half_gaussian(3,1)
dg = np.concatenate([-g,np.flip(g[0:-1])])
g = make_half_gaussian(3,2)
ddg = np.concatenate([g,np.flip(g[0:-1])])

w = len(ddg) // 2
x = np.array(range(-w,w+1))

np.dot(dg,np.ones(len(dg)))
np.dot(dg,x)
np.dot(dg,np.power(x,2))

np.dot(ddg,np.ones(len(ddg)))
np.dot(ddg,x)
np.dot(ddg,np.power(x,2))



plt.plot(ddg)
plt.show()


def make_half_gaussian(sigma, derivative_order):
	half_gaussian_size = 9
	half_filter_size = 1 + half_gaussian_size

	filter = np.zeros(half_filter_size)
	r0 = half_filter_size - 1
	sigma2 = sigma * sigma

	if derivative_order == 0:
		factor = -0.5 / sigma2
		normalization = 0
		filter[r0] = 1.0
		for rr in range(1, half_filter_size):
			rad = float(rr)
			g = math.exp(factor * (rad * rad))
			filter[r0 - rr] = g
			normalization += g
		normalization = 1.0 / (normalization * 2 + 1)
		for rr in range(half_filter_size):
			filter[rr] *= normalization

	elif derivative_order == 1:
		factor = -0.5 / sigma2
		moment = 0.0
		filter[r0] = 0.0
		for rr in range(1, half_filter_size):
			rad = float(rr)
			g = rad * math.exp(factor * (rad * rad))
			filter[r0 - rr] = g
			moment += rad * g
		normalization = 1.0 / (2.0 * moment)
		for rr in range(half_filter_size - 1):
			filter[rr] *= normalization

	elif derivative_order == 2:
		norm = 1.0 / (math.sqrt(2.0 * math.pi) * sigma * sigma2)
		mean = 0.0
		filter[r0] = -norm
		for rr in range(1, half_filter_size):
			rad = float(rr)
			sr2 = rad * rad / sigma2
			g = (sr2 - 1.0) * norm * math.exp(-0.5 * sr2)
			filter[r0 - rr] = g
			mean += g
		mean = (mean * 2.0 + filter[r0]) / (float(r0) * 2.0 + 1.0)  # mean of all values in filter
		filter[r0] -= mean
		moment = 0.0
		for rr in range(1, half_filter_size):
			rad = float(rr)
			filter[r0 - rr] -= mean
			moment += rad * rad * filter[r0 - rr]
		normalization = 1.0 / moment
		for rr in range(half_filter_size):
			filter[rr] *= normalization

	elif derivative_order == 3:
		norm = 1.0 / (math.sqrt(2.0 * math.pi) * sigma * sigma2 * sigma2)
		filter[r0] = 0.0
		moment = 0.0
		for rr in range(1, half_filter_size):
			rad = float(rr)
			rr2 = rad * rad
			sr2 = rr2 / sigma2
			g = norm * math.exp(-0.5 * sr2) * (rad * (3.0 - sr2))
			filter[r0 - rr] = g
			moment += g * rr2 * rad
		normalization = 3.0 / moment
		for rr in range(half_filter_size):
			filter[rr] *= normalization

	else:
		raise NotImplementedError("Derivative order not implemented")

	return filter
