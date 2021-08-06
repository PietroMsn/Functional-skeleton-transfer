import numpy as np
from pyflann import *
import scipy as sp
from scipy.sparse.linalg import eigsh
import tensorflow as tf
from goodname.WKM import WKM
import robust_laplacian as rl



def compute_phi(m_v, m_f, k, isPC):


    '''WSrc, Area_M = computeAB(v_M, f_M)
    WTar, Area_N = computeAB(v_N, f_N)
    evals_M, evecs_M = eigsh(WSrc, 400, Area_M, sigma=0.0, which='LM')
    evals_N, evecs_N = eigsh(WTar, 400, Area_N, sigma=0.0, which='LM')'''

    if isPC:
        L, M = rl.point_cloud_laplacian(m_v)

    else:
        m_f = m_f.astype(int)
        L, M = rl.mesh_laplacian(m_v, m_f)

    try:
        evals, evecs = eigsh(L, k, M, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
    except:
        evals, evecs = eigsh(L - 1e-8 * sp.sparse.identity(m_v.shape[0]), k,
                                     M, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)

    return evals, evecs, M


def get_coeff(v, f, k, isPC):

    evals, evecs, Mass = compute_phi(v, f, k, isPC)

    # convert into a sparse matrices multiplication to prevent Out of memory exception
    sparse_evecs = sp.sparse.csr_matrix(evecs[:, 0:k].T)
    pinv_phi = sparse_evecs.dot(Mass) #np.asarray(evecs[:, 0:k].T, dtype=float32) @ np.asarray(Mass.todense(), dtype=float32)

    #id = np.matmul(pinv_phi, phi)
    #print(id)

    return evals, evecs, Mass, np.asarray(pinv_phi.todense())


'''def get_C_corresp(pinv_phi1, phi2, nvert):
    Id = np.identity(nvert)

    return np.matmul(np.matmul(pinv_phi1, Id), phi2)'''

def WKS(vertices, evals, evecs, wks_size, variance):
    # Number of vertices
    n = vertices.shape[0]
    WKS = np.zeros((n, wks_size))

    # Just for numerical stability
    evals[evals < 1e-6] = 1e-6

    # log(E)
    log_E = np.log(evals).T

    # Define the energies step
    e = np.linspace(log_E[1], np.max(log_E) / 1.02, wks_size)

    # Compute the sigma
    sigma = (e[1] - e[0]) * variance
    C = np.zeros((wks_size, 1))

    for i in np.arange(0, wks_size):
        # Computing WKS
        WKS[:, i] = np.sum(
            (evecs) ** 2 * np.tile(np.exp((-(e[i] - log_E) ** 2) / (2 * sigma ** 2)), (n, 1)), axis=1)

        # Noramlization
        C[i] = np.sum(np.exp((-(e[i] - log_E) ** 2) / (2 * sigma ** 2)))

    WKS = np.divide(WKS, np.tile(C, (1, n)).T)
    return WKS

'''def WKS_simone(vertices, Phi, Lambda, wks_variance, wks_size=100):

    nv = np.shape(vertices)[0]

    # compute time scales
    log_E = np.log(np.maximum(Lambda, 1e-6))
    e = np.linspace(log_E[1], (np.max(log_E)) / 1.02, num=wks_size)
    sigma = (e[2] - e[1]) * wks_variance

    WKS = np.zeros((nv, wks_size))
    ratio = np.zeros((1, wks_size))

    for i in range(0, wks_size):
        WKS[:, i] = np.sum(
            np.square(Phi) * np.tile(np.transpose(np.exp(-np.square((e[i] - log_E)) / (2 * np.square(sigma)))),
                                     (nv, 1)), 1);
        ratio[0, i] = np.sum(np.exp(-np.square((e[i] - log_E)) / (2 * np.square(sigma))))

    WKS = WKS / np.tile(ratio, (nv, 1))
    return WKS'''

'''def WKM_sione(vertices, landmarks, Phi, Lambda, wks_size=100):

    nv = len(vertices)
    n_evecs = 400
    WKM = np.zeros((nv, wks_size * len(landmarks)))

    for li in range(0, len(landmarks)):
        segment = np.zeros((nv, 1))
        segment[int(landmarks[li])] = 1

        numEigenfunctions = n_evecs

        absoluteEigenvalues = np.abs(Lambda)
        emin = np.log(absoluteEigenvalues[1])
        emax = np.log(absoluteEigenvalues[len(absoluteEigenvalues) - 1])
        s = 7 * (emax - emin) / wks_size
        emin = emin + 2 * s
        emax = emax - 2 * s
        es = np.linspace(emin, emax, wks_size)

        T = np.exp(-np.square(np.tile(np.log(absoluteEigenvalues), (1, wks_size)) - np.tile(es.T, (n_evecs, 1))) / (
                    2 * np.square(s)))
        wkm = T * np.tile(np.dot(Phi.T, segment), (1, np.shape(T)[1]))
        wkm = np.dot(Phi, wkm)

        WKM[:, wks_size * li:wks_size * (li + 1)] = wkm

    return WKM'''


def get_map(v_M, f_M, evals_M, evecs_M, M_M, v_N, f_N, evals_N, evecs_N, M_N, k1, k2, dim_out, landM, landN, params):

    sub_sample = 10
    n_evalsM = k1
    n_evalsN = k2
    n_wks = 200

    # Landmarks
    landmarks2 = landN
    landmarks1 = landM

    d_M = WKS(v_M, evals_M, evecs_M, n_wks, 7)
    d_N = WKS(v_N, evals_N, evecs_N, n_wks, 7)

    # WKM
    land1 = WKM(v_M, evals_M, evecs_M, landmarks1, n_wks)
    land2 = WKM(v_N, evals_N, evecs_N, landmarks2, n_wks)

    # Optimization Process
    desc_M = np.hstack((d_M, land1))
    desc_N = np.hstack((d_N, land2))

    # Subsampling of the descriptors
    desc_M = desc_M[:, ::sub_sample]
    desc_N = desc_N[:, ::sub_sample]


    # Descriptor normalization
    no = np.sqrt(np.diag(np.matmul(M_M.T.__matmul__(desc_M).T, desc_M)))
    no_M = np.tile(no.T, (v_M.shape[0], 1))
    no_N = np.tile(no.T, (v_N.shape[0], 1))
    fct_M = np.divide(desc_M, no_M)
    fct_N = np.divide(desc_N, no_N)
    #fct_N = np.asarray((sp.sparse.csr_matrix(desc_N) / sp.sparse.csr_matrix(no_N)))

    # Coefficents of the obtained descriptors
    Fct_M = np.matmul(M_M.T.__matmul__(evecs_M[:, 0:n_evalsM]).T, fct_M)
    Fct_N = np.matmul(M_N.T.__matmul__(evecs_N[:, 0:n_evalsN]).T, fct_N)


    # The relation between the two constant functions can be computed in a closed form
    constFct = np.zeros((n_evalsM, 1))
    constFct[0, 0] = np.sign(evecs_M[0, 0] * evecs_N[0, 0]) * np.sqrt(np.sum(M_N) / np.sum(M_M))

    # Different way to compute Laplacian commutativity
    # Dlb = (np.tile(evals_src[0:n_evals], (n_evals, 1)) - np.tile(evals_tar[0:n_evals].T, (n_evals, 1)))**2
    # Dlb = np.float32(Dlb/tf.reduce_sum((Dlb**2)))

    # Energy weights
    a = 1e-1  # Descriptors preservation
    c = 1e-8  # Commutativity with Laplacian

    # Define tensorflow objects
    fs = tf.constant(Fct_M, dtype=tf.float32)
    ft = tf.constant(Fct_N, dtype=tf.float32)
    evalM = tf.constant(tf.linalg.tensor_diag(np.reshape(np.float32(evals_M[0:n_evalsM]), (n_evalsM,))),
                        dtype=tf.float32)
    evalN = tf.constant(tf.linalg.tensor_diag(np.reshape(np.float32(evals_N[0:n_evalsN]), (n_evalsN,))),
                        dtype=tf.float32)

    # Initialize C
    C_ini = np.zeros((n_evalsN, n_evalsM))
    C_ini[0, 0] = constFct[0, 0]
    C = tf.Variable(tf.zeros((n_evalsN, n_evalsM), dtype=tf.float32))
    C.assign(C_ini)

    # Optimizer
    adam = tf.keras.optimizers.Adam(1e-1)  # Optimization technique
    trainable_vars = [C]

    # Optimization
    for i in np.arange(0, 1000):
        with tf.GradientTape() as tape:
            loss1 = a * tf.reduce_sum(((tf.matmul(C, fs) - ft) ** 2)) / 2  # Descriptor preservation
            loss2 = c * tf.reduce_sum((tf.matmul(C, evalM) - tf.matmul(evalN,
                                                                       C)) ** 2)  # tf.reduce_sum(((C ** 2) * Dlb) / 2)  # Commute with Laplacian
            # loss3 = 1e-6 *  tf.reduce_sum(tf.square(tf.matmul(tf.transpose(C),C) - tf.eye(n_evals))) # Orthonormal C
            loss = loss1 + loss2  # + loss3

        # Apply gradient
        grad = tape.gradient(loss, trainable_vars)
        tmp = adam.apply_gradients(zip(grad, trainable_vars))

    C_ICP = C.numpy()


    flann = FLANN()
    print('ICP refine...')
    for k in np.arange(0, 5):
        result, dists = flann.nn((C_ICP@evecs_M[:, 0:n_evalsM].T).T, evecs_N[:, :n_evalsN], 1)
        W = np.linalg.lstsq(evecs_N[:, 0:n_evalsN], evecs_M[result, 0:n_evalsM])[0]
        d = np.linalg.svd(W)
        C_ICP = np.matmul(np.matmul(d[0], np.eye(n_evalsN, n_evalsM)), d[2])


    result_ini = result

    N_eigs_final = dim_out
    N_eigs_init = k2
    step = 10

    # Convert it into a Functional Maps
    C_iter = np.matmul(np.linalg.pinv(evecs_N[:, 0:N_eigs_init]), evecs_M[result_ini, 0:N_eigs_init])

    # ZoomOut
    for i in np.arange(0, N_eigs_final - N_eigs_init, step):
        # 1) Convert into dense correspondence
        #evecs_transf = np.matmul(evecs_M[:, 0: i], C_iter.T)
        evecs_transf = np.matmul(C_iter, evecs_M[:, 0:N_eigs_init + i].T).T
        #result, dists = flann.nn(evecs_transf, evecs_N[:, 0:i], 1)
        result, dists = flann.nn(evecs_transf, evecs_N[:, 0:N_eigs_init + i], 1)

        # 2) Convert into C of dimension (n+1) x (n+1)
        #C_iter = np.matmul(np.linalg.pinv(evecs_N[:, 0:i + step]), evecs_M[result, 0:i + step ])
        C_iter = np.matmul(np.linalg.pinv(evecs_N[:, 0:N_eigs_init + i + step]), evecs_M[result, 0:N_eigs_init + i + step])

    return C_iter, result
