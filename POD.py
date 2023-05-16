import numpy as np


#%  load R1cut.mat
#load data.mat
#% load coordinates.mat

# if isa(x1,'cell')       % to convert any cell type to double
#     x1 = str2double(x1);
#     y1 = str2double(y1);
# end


data = D                  # for use later in phase averaging
ind = np.isnan[D[:,1]]  #  locate NaN rows
loc = ind.find("1")
D = D[loc, :]           # eliminate NaN rows (to be inserted back later)

mid = np.numel(x1)         # total number of data points in one frame
M = np.mean(D,2)
D = D - np.repmat(M, 1, np.size(D,2))  # eliminating mean fron the data set

# POD algorithm

CorMat = (np.transpose(D)*D) #/(size(D,2));       % corelation matrix
[Ev,Eg] = np.eig(CorMat)              # eigen decomposition
L = np.diag(Eg)                  # extracting diagonal vales from eigen value matrix
[EigVal,P] = np.sort(L,'descend')     # sorting eigen values in descending order
EigVal[end] = 0                    # assigning last eigen value equal to zero
Evec = Ev[:,P]                 # sorting eigen vectors corresponding to eigen velues
Phi = -(D*Evec)*(np.repmat(np.transpose((EigVal)^-0.5), np.size(D,1),1))   # POD mode determination
TimeCoeff = np.transpose(Phi)*D               # time coefficients (rows)
Reconstruction = Phi[:, 1:5]*TimeCoeff[1:5,:]     # reconstruction of flow field using finite number of POD modes 

#end of POD algorithm

tmp1 = np.nan(np.numel(x1)*2, np.size(D,2))    
tmp1[loc,:] = Phi      # inserting back NaN rows in POD modes matrix
Phi = tmp1             # inserting back NaN rows in POD modes matrix
tmp = np.nan(np.numel(x1)*2, 1)
tmp[loc] = M           # inserting back NaN rows in Mean velocity array
M = tmp                # inserting back NaN rows in Mean velocity array