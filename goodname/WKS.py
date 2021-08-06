def WKS(vertices, Lambda, Phi, Area, wks_size = 100):
    import numpy as np
    nv = np.shape(vertices)[0]
    wks_variance = 6;


    # compute time scales
    log_E = np.log(np.maximum(Lambda, 1e-6))
    e = np.linspace(log_E[1], (np.max(log_E)) / 1.02, num=wks_size)
    sigma = (e[2] - e[1]) * wks_variance

    WKS = np.zeros((nv, wks_size))
    ratio = np.zeros((1, wks_size))

    for i in range(0, wks_size):
        WKS[:, i] = np.sum(
            np.square(Phi) * np.tile(np.transpose(np.exp(-np.square((e[i] - log_E)) / (2 * np.square(sigma)))),
                                     (nv, 1)), 1)
        ratio[0, i] = np.sum(np.exp(-np.square((e[i] - log_E)) / (2 * np.square(sigma))))

    WKS = WKS / np.tile(ratio, (nv, 1))
    return WKS
