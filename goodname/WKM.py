def WKM(vertices, Lambda, Phi, landmarks, wks_size = 100):
    import numpy as np

    n_evecs = len(Phi[0, :])
    nv = np.shape(vertices)[0]

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
    
        T = np.exp(-np.square(np.tile(np.reshape(np.log(absoluteEigenvalues),(n_evecs,1)), (1, wks_size)) - np.tile(es.T, (n_evecs, 1))) / (
                    2 * np.square(s)))
        wkm = T * np.tile(np.dot(Phi.T, segment), (1, np.shape(T)[1]))
        wkm = np.dot(Phi, wkm)
    
        WKM[:, wks_size * li:wks_size * (li + 1)] = wkm

    return WKM
