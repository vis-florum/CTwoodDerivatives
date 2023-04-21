import numpy as np
from scipy import ndimage
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from matplotlib.pyplot import quiver
import diplib as dip


## Artificial Image
M = 100
N = 100
r = N/3	# radius of circle
delta = r/7 # transition width around circle
growthRingWidth = 10 # pixels
tanWaveStrength = 0.02 # strength of tangential wavyness
tanWavePeaks = 5 # number of peaks in tangential wavyness

I = np.zeros((M,N))

## Homogeneous Circle
mx = int(np.ceil(N/2))
my = int(np.ceil(M/2))
x,y = np.meshgrid(np.arange(1,N+1),np.arange(1,M+1))
R = np.zeros((M,N,2))
R[:,:,0] = (mx - x)
R[:,:,1] = (my - y)
RR = np.hypot(R[:,:,0],R[:,:,1])
I[RR < r] = 1
transition = (RR >= r) & (RR < r+delta)
I[transition] = np.exp((r - RR[transition])/2)


## GST Analysis using Diplib
EV = dip.StructureTensor(I,gradientSigmas=[1.],tensorSigmas=[1.])
# EV is array of symmetric matrices ergo vector is simply [(1,1), (1,2), (2,2)] of matrix
outs = ["l1", "l2", "orientation", "energy", "anisotropy1", "anisotropy2", "curvature"]
EVA = dip.StructureTensorAnalysis(EV,outs)
I_g = dip.Gauss(I, 10.0)

# Plot defaults
plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['figure.origin'] = 'upper'

# Montage plot
plt.figure()
plt.subplot(2, 4, 1)
plt.imshow(I)
plt.title("original")
plt.axis('image')
for i in range(len(outs)):
    plt.subplot(2, 4, i+2)
    plt.imshow(EVA[i])
    plt.title(outs[i])
    plt.axis('image')
plt.show()


# Convert orientations to 2D vector field and plot it in a quiver plot
plt.figure()
t = 3  # plot only every t-th arrow in a quiver plot
x, y = np.meshgrid(np.arange(I.shape[1]), np.arange(I.shape[0]))
X = x[::t, ::t]
Y = y[::t, ::t]
M = EVA[3][::t, ::t] > 0.001  # scaling the arrows
U =  np.cos(EVA[2])[::t, ::t]*M
V = -np.sin(EVA[2])[::t, ::t]*M  # if using imshow, flip y axis
plt.imshow(I)
plt.quiver(X, Y, U, V, scale=30, color='r')
plt.show()
plt.imsave('vectors_diplib.png',I) # export as png





# def gkern(l=5, sig=1.):
#     """\
#     creates gaussian kernel with side length `l` and a sigma of `sig`
#     """
#     ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
#     gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
#     kernel = gauss # np.outer(gauss, gauss)
#     return kernel / np.sum(kernel)

# def gst_2d_v2(I, sigma1=1, sigma2=3, thresh=0.8):
    
#     n, m = I.shape
    
#     cutoff = int(np.ceil(3 * sigma1))

#     h = gkern(2 * cutoff + 1, sigma1)
#     h_grad = np.gradient(h, axis=0)

#     gx = ndimage.convolve1d(I, h_grad, mode='mirror')
#     gy = ndimage.convolve1d(I, h_grad.T, mode='mirror')

#     H = np.zeros((n, m, 2, 2))
#     H[..., 0, 0] = gaussian(gx ** 2, sigma2)
#     H[..., 0, 1] = gaussian(gx * gy, sigma2)
#     H[..., 1, 1] = gaussian(gy ** 2, sigma2)
#     H[..., 1, 0] = H[..., 0, 1]

#     c1 = np.zeros((n, m))
#     e1 = np.zeros((n, m, 2))
#     e2 = np.zeros((n, m, 2))
#     Hmat = np.zeros((2, 2))

#     for i in range(n):
#         for j in range(m):
#             Hmat[:, :] = H[i, j, :, :]
#             U, s, vh = np.linalg.svd(Hmat)
#             c1[i, j] = (U[0, 0] - U[1, 1]) ** 2 / (U[0, 0] ** 2 + U[1, 1] ** 2)
#             e1[i, j, :] = U[:, 0]
#             e2[i, j, :] = U[:, 1]

#     e1[c1 <= thresh] = 0
#     e2[c1 <= thresh] = 0

#     return e1, e2