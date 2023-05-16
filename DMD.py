import numpy as np

dd = dir('*.mat')
t1 = 1
t2 = np.numel(dd)

for k in (t1, t2):
        #load(['R' num2str(k) '.mat']);    # edit input name accordingly
        if (k==1):
            D1 = D[:,1:end-1]
            D2 = D[:,2:end]
        else:
            D1 = (D1, D[:,1:end-1])
            D2 = (D2, D[:,2:end])

# numofruns = t2
# %% use this section in case of single runs
# % clear all
# % load data
# % load coordinates
# % numofruns = 1;
# %%
# if isa(x1,'cell')       # to convert any cell type to double
#     x1 = float(x1)
#     y1 = float(y1)
# end

Frate = 2400   # Frame rate  (data acquisition)
TruncVal = 3000     # truncation value

# DMD algorithm

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

# end of DMD algorithm

modeNorm = np.zeros(1,TruncVal)
for i in (1, TruncVal):
    modeNorm[i] = np.linalg.norm(X[:,i]) # norm of each mode

mag = np.absolute(eivarray)
weigh = mag[1:end]/np.sum(mag)
delta_t = 1/Frate
omega = np.log(eivarray)/delta_t
growth = np.real(omega)
frequency = np.imag(omega)/(2*np.pi)