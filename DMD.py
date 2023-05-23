import numpy as np

data = np.array([
        [1, 2, 3, 4, 5],
        [5, 6, 7, 8, 10],
        [5, 6, 7, 8, 10],
        [5, 6, 7, 8, 10],
        [5, 6, 7, 8, 10],
        ])

data_shape = data.shape
percent_Trunc = 0.03
TruncVal = data_shape[0]*percent_Trunc

#%% Compression matrix generation

C_gaussian = np.random.normal(0, 1, data_shape)

#%% Compress data

data = np.dot(C_gaussian, data)

# %% DMD

[U, S, V] = np.linalg.svd(data, full_matrices=True, compute_uv=True, hermitian=False)
U = U[:, 1:TruncVal]
S = S[1:TruncVal,1:TruncVal]
V = V[:,1:TruncVal]
Atilde = np.transpose(U) * D2 * V / S
[eivVec,eivVal] = np.linalg.eig(Atilde)
eivarray = np.diag(eivVal)
X1 = (D2 * V / (S)* eivVec)/np.matlib.repmat(np.transpose(eivarray),np.size(D1,1),1)  # unscaled modes
d = X1/D1[:,1]  # scales
amplitude = np.absolute(d)
X=X1*np.diag(d)    #scaled mode

# end of DMD

# %%
modeNorm = np.zeros(1,TruncVal)
for i in (1, TruncVal):
    modeNorm[i] = np.linalg.norm(X[:,i]) # norm of each mode

mag = np.absolute(eivarray)
weigh = mag[1:-1]/np.sum(mag)
delta_t = 1/frame_rate   
omega = np.log(eivarray)/delta_t
growth = np.real(omega)
frequency = np.imag(omega)/(2*np.pi)

# %%
freq1 = np.where(frequency>=0)     # eliminating negative frequencies
amplitude1 = amplitude[freq1]  # eliminating amplitudes corresponding to -ve frequencies
d1 = d[freq1]    # similar operations below for different parameters
omega1 = omega[freq1]
growth1 = growth[freq1]
weigh1 = weigh[freq1]
modeNorm1 = modeNorm[:,freq1]
frequency1 = frequency[freq1]
Xplus = X[:,freq1]