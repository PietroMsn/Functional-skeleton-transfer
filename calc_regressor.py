import numpy as np
from spatial_regressor import regressor_single_joint_spatial
from functional_maps import get_map


''' This function calls both the optimization of the spectral and spatial regressor 
returns:
- Joints calculated with the spectral regressor
- phi for the target shape
- pseudoinverse fot the target shape
- the filter F for the selected vertices (the mask to localize the optimization)
- the joints calculated using the regressor projected from spatial to spectral domain
'''
def get_joints(JointsX, X, Y, faces_X, faces_Y, weights, parents, k1, k2, landSrc, landTar, params, L1, L2):

    evalsX, phiX, MX, pinv_phiX = L1['evalsX'], L1['phiX'], L1['MX'], L1['pinv_phiX']
    evalsY, phiY, MY, pinv_phiY = L2['evalsY'], L2['phiY'], L2['MY'], L2['pinv_phiY']

    MX_sum = MX.sum()
    MY_sum = MY.sum()
    scale = np.sqrt(MX_sum / MY_sum)
    Y = np.multiply(scale, Y)

    Y_hat = np.matmul(pinv_phiY[:k2, :], Y)
    X_hat = np.matmul(pinv_phiX[:k1, :], X)


    kZO1, kZO2 = params['kZO1'], params['kZO2']
    dim_out = k1
    C, matchesCalc = get_map(Y, faces_Y.astype(int), evalsY, phiY, MY, X, faces_X.astype(int), evalsX, phiX, MX, kZO2, kZO1,
                       dim_out, landTar, landSrc, params)

    # calculates the spatial regressor (best results for now)
    R_spatial, F_spatial = regressor_single_joint_spatial(JointsX, X, weights, parents, params)

    R_hat_from_spatialSrc = np.matmul(R_spatial, phiX[:, :k1])

    return R_spatial, R_hat_from_spatialSrc, phiX, phiY, pinv_phiY, pinv_phiX, np.asarray(X_hat), np.asarray(Y_hat), Y, C, F_spatial, matchesCalc








