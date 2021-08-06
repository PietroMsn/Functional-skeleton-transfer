import pyvista as pv
import numpy as np
from pyflann import *


# from n x 3 faces array to polydata face (concatenated indices )
def arraytofaces(arr):
    res = np.zeros((len(arr)*4))
    for i in range(len(arr)):
        res[i * 4] = 3 #the first number of each polygon is the number of verts
        res[i * 4 + 1] = arr[i, 0]
        res[i * 4 + 2] = arr[i, 1]
        res[i * 4 + 3] = arr[i, 2]
    return np.asarray(res, dtype=int)


def displaySpectralRes(Data, jointsSrc, vSrc, fSrc, fTar, jointsTar, jointsTarRec, vTar, F, params):
    SMPLRegressor = np.array(Data['J_regressor'])
    jointsOriginalSrc = jointsSrc#np.matmul(SMPLRegressor, vSrc)


    pointsOriginalJSrc = pv.PolyData(jointsOriginalSrc)

    shapeSrc = pv.PolyData(vSrc, np.asarray(fSrc, dtype=int))

    pointsJTar = pv.PolyData(jointsTar)
    shapeTar = pv.PolyData(vTar,  np.asarray(fTar, dtype=int))

    pointsJTarReconstr = pv.PolyData(jointsTarRec)

    # plot the source mesh with source joints
    p = pv.Plotter(shape=(1, 3), border=False)
    p.subplot(0, 0)
    p.add_mesh(shapeSrc, opacity=0.35)
    p.add_mesh(pointsOriginalJSrc, 'r')
    # Draw the skeleton
    for i in range(len(jointsOriginalSrc)):
        parents = Data['kintree_table'][0, :]
        parents[0] = 0
        line = pv.Line(jointsOriginalSrc[parents[i], :], jointsOriginalSrc[i, :])
        p.add_mesh(line, color='b')

    '''p.add_mesh(jointsSrc + P[0], 'y')
    p.add_mesh(jointsSrc + P[1], 'g')
    p.add_mesh(jointsSrc + P[2], 'b')'''
    p.add_text("Original SMPL Joints", font_size=10)


    # plot the target mesh with evaluated joints
    p.subplot(0, 1)
    p.add_mesh(shapeTar, opacity=0.35)
    p.add_mesh(pointsJTar, 'r')
    # Draw the skeleton
    for i in range(len(jointsTar)):
        parents = Data['kintree_table'][0, :]
        parents[0] = 0
        line = pv.Line(jointsTar[parents[i], :], jointsTar[i, :])
        p.add_mesh(line, color='b')

        indJoint = pv.PolyData(jointsTar[18, :])
        p.add_mesh(indJoint, 'g')
    #p.add_mesh(jointsTar + P_transfer, 'y')
    '''p.add_mesh(jointsSrc + P_transfer[0], 'y')
    p.add_mesh(jointsSrc + P_transfer[1], 'g')
    p.add_mesh(jointsSrc + P_transfer[2], 'b')'''

    p.add_text("Spectral Regressor on Target", font_size=10)

    # plot the target mesh with evaluated joints
    p.subplot(0, 2)
    if params['isPC2']:
        p.add_mesh(pv.PolyData(vTar), opacity=0.35, point_size=2)
    else:
        p.add_mesh(shapeTar, opacity=0.35)
    p.add_mesh(pointsJTarReconstr, 'r')
    # Draw the skeleton
    for i in range(len(jointsTar)):
        parents = Data['kintree_table'][0, :]
        parents[0] = 0
        line = pv.Line(np.squeeze(jointsTarRec[parents[i], :]), np.squeeze(jointsTarRec[i, :]))
        p.add_mesh(line, color='b')

        indJoint = pv.PolyData(jointsTarRec[18, :])
        p.add_mesh(indJoint, 'g')


    p.add_text("Spatial Regressor On Target", font_size=10)
    p.show()
