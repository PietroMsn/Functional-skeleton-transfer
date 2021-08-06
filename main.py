'''
A functional skeleton transfer
CODE for transferring a skeleton from one shape to another
'''

import scipy.io as sio
from calc_regressor import get_joints
from extra_modules import displaySpectralRes, arraytofaces
import numpy as np



params = {}

# if the target shape is a point cloud these values should be put to True
params['isPC1'] = False
params['isPC2'] = False

k1, k2 = 120, 120                           # set number of eigenfunction for source and target respectively
params['kZO1'], params['kZO2'] = 20, 20     # set number of eigenfunction for source and target before zommOut

source = sio.loadmat('./data/basicmodel_m_lbs.mat')

target = sio.loadmat('./data/50007_knees_001806.mat')

# storing the hierarchy of SMPL skeleton
parents = np.transpose(source['kintree_table'][0, :])
parents[0] = 0

# the 7 SMPL landmarks
landSrc = (np.asarray([3509, 3076, 414, 5909, 6683, 3216, 2444], dtype=int) - 1).T

# preaparing the source shape
fSrcArray = np.asarray(source['f'], dtype=int)
vSrc = np.asarray(source['v_template'])
fSrc = arraytofaces(fSrcArray)

#storing the source Joints
J_regressor = source['J_regressor'].todense()
J = np.matmul(np.asarray(J_regressor), vSrc)
jointsSrc = J
weights = source['weights']

jointsTar = np.zeros((len(J), 3))

#prepearing the target shape and its 7 landmarks
vTar = target['vertex']
fTarArray = np.asarray(target['faces'], dtype=int) - 1
landTar = np.squeeze(np.asarray(target['landmarkRes'], dtype=int) - 1)

# computation of the eigenfunction and the eigenvalues of the two shapes
from functional_maps import get_coeff

evalsX, phiX, MX, pinv_phiX = get_coeff(vSrc, fSrcArray, 400, params['isPC1'])
evalsY, phiY, MY, pinv_phiY = get_coeff(vTar, fTarArray, 400, params['isPC2'])
AX = MX.sum()
AY = MY.sum()
scale = np.sqrt(AX / AY)
vTar = np.multiply(scale, vTar)
fTar = arraytofaces(fTarArray)

L1 = {'evalsX': evalsX, 'phiX': phiX, 'MX': MX, 'pinv_phiX': pinv_phiX}
L2 = {'evalsY': evalsY, 'phiY': phiY, 'MY': MY, 'pinv_phiY': pinv_phiY}

# computation of the functional map and the skeleton regressor
R_spatial, R_hat_from_spatial, phiSrc, phiTar, pinv_phiTar, pinv_phiSrc, Src_hat, Tar_hat, vTar, C, F, \
matches = get_joints(J, vSrc, vTar, fSrcArray, fTarArray, weights, parents, k1, k2, landSrc, landTar, params, L1, L2)

# mathces is the point to point map obtained from the map
mapped_vTar = np.squeeze(vTar[matches, :])
Jtar1 = np.matmul(np.matmul(R_hat_from_spatial, np.array(C)), Tar_hat)  # Transfer of joints through the map C
Jtar2 = np.matmul(R_spatial, vTar[matches, :])                          # Transfer of joints through the p2p map

# Plot results
displaySpectralRes(source, jointsSrc, vSrc, fSrc, fTar, Jtar1, Jtar2, vTar, F, params)





