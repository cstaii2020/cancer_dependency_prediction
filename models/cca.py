import numpy as np
import scipy.stats as ss


def scale(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def scale_train_test(train_data, test_data):
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    return (train_data - mean) / std, (test_data - mean) / std


def unscale_prediction(train_data, predictions):
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    return (std * predictions) + mean


def cca_gamma(x, y):
    """
    Canonical Correlation Analysis
    Currently only returns the canonical correlations.
    """
    n, p1 = x.shape
    n, p2 = y.shape
    x, y = scale(x), scale(y)

    Qx, Rx = np.linalg.qr(x)
    Qy, Ry = np.linalg.qr(y)

    rankX = np.linalg.matrix_rank(Rx)
    if rankX == 0:
        raise Exception('Rank(X) = 0! Bad Data!')
    elif rankX < p1:
        # warnings.warn("X not full rank!")
        Qx = Qx[:, 0:rankX]
        Rx = Rx[0:rankX, 0:rankX]

    rankY = np.linalg.matrix_rank(Ry)
    if rankY == 0:
        raise Exception('Rank(X) = 0! Bad Data!')
    elif rankY < p2:
        # warnings.warn("Y not full rank!")
        Qy = Qy[:, 0:rankY]
        Ry = Ry[0:rankY, 0:rankY]

    d = min(rankX, rankY)
    svdInput = np.dot(Qx.T, Qy)

    U, r, V = np.linalg.svd(svdInput)
    r = np.clip(r, 0, 1)
    # A = np.linalg.lstsq(Rx, U[:,0:d]) * np.sqrt(n-1)
    # B = np.linalg.lstsq(Ry, V[:,0:d]) * np.sqrt(n-1)

    # TODO: resize A to match inputs

    # return (A,B,r)
    return r, 0, 0


def tutorial_on_cca(a, b):
    """
    Solving CCA through the Standard Eigenvalue Problem, pages 4-6 in
    https://mycourses.aalto.fi/pluginfile.php/480042/course/section/97794/uurtio_et_al_2017_final.pdf
    :param a:
    :param b:
    :return:
    """

    # getting parameters and scaling
    n, p, q = a.shape[0], a.shape[1], b.shape[1]
    m = min(p, q)
    a, b = scale(a), scale(b)

    # computing covariance matrices, and joining it to the joint covariance matrix - equation 2
    caa = np.dot(a.T, a) / (n - 1)
    cab = np.dot(a.T, b) / (n - 1)
    cba = np.dot(b.T, a) / (n - 1)
    cbb = np.dot(b.T, b) / (n - 1)

    # if cbb and caa is invertible
    if np.linalg.det(cbb*1000) != 0 and np.linalg.det(caa*1000) != 0:
        temp = np.dot(np.linalg.inv(caa), cab)

        # equation 10 - 11
        rho_squared, eigen_vecs = np.linalg.eig(np.linalg.multi_dot((np.linalg.inv(cbb), cba, temp)))
        rho = np.sqrt(rho_squared[:m].real)
        wb = eigen_vecs[:, :m].real

        # equation 10
        wa = np.dot(temp, wb) / rho
        return rho, wa, wb
    else:
        raise ValueError('caa or cbb not invertible, not implemented yet!')


def generalized_eigenvalue(a, b):
    # getting parameters and scaling
    n, p, q = a.shape[0], a.shape[1], b.shape[1]
    m = min(p, q)
    # a, b = scale(a), scale(b)

    # computing covariance matrices, and joining it to the joint covariance matrix - equation 2
    caa = np.dot(a.T, a) / (n - 1) # (p*p)
    cab = np.dot(a.T, b) / (n - 1) #(p*q)
    cba = np.dot(b.T, a) / (n - 1) # (q*p)
    cbb = np.dot(b.T, b) / (n - 1) # (q*q)

    cabba = np.block([[np.zeros((p, p)), cab],
                      [cba, np.zeros((q, q))]])
    caabb = np.block([[caa, np.zeros((p, q))],
                      [np.zeros((q, p)), cbb]])

    alpha = np.sort(get_positives(np.linalg.eigvals(cabba).real))[::-1]
    betta = np.sort(get_positives(np.linalg.eigvals(caabb).real))[::-1]

    rho = np.sort(alpha[:m] / betta[:m])[::-1]

    wa = np.zeros((p, m))
    wb = np.zeros((q, m))
    for i in range(m):
        wawb = np.linalg.solve(cabba - rho[i]*caabb, np.zeros(p+q))
        wa[:, i] = wawb[:p]
        wb[:, i] = wawb[p:]

    return rho[:m], wa, wb


def evaluate_cca(a, b, gamma=False):
    n, p, q = a.shape[0], a.shape[1], b.shape[1]
    m = min(p, q)

    if gamma:
        rho, wa, wb = cca_gamma(a, b)
    else:
        rho, wa, wb = tutorial_on_cca(a, b)

    l = []
    for i in range(m):
        current_l = -(n - i - 0.5 * (p + q + 1))
        current_l -= np.sum(np.power(rho[:i], -2))
        current_l *= np.sum(np.log(1 - np.square(rho[i:])))
        l.append(current_l)

    p_vals = []
    dofs = []
    for j in range(m):
        dof = (p - j) * (m - j)
        dofs.append(dof)
        current_chi = ss.chi2(dof)
        p_vals.append(1 - current_chi.cdf(l[j]))

    # Empirical p value
    # empirical_tests = 100000
    # better_rho_count = np.zeros(len(rho))
    # for test_count in range(empirical_tests):
    #     print("%d/%d -> %.1f%% test"%(test_count, empirical_tests, test_count/empirical_tests*100))
    #     rho_random, _, _ = tutorial_on_cca(np.random.permutation(a), b)
    #     for i, r in enumerate(rho_random):
    #         if r > rho[i]:
    #             better_rho_count[i] += 1
    #
    # empirical_p_values = (better_rho_count + 1) / empirical_tests
    return rho, p_vals


def evaluate_cca_wa_wb(a, b):
    n, p, q = a.shape[0], a.shape[1], b.shape[1]
    m = min(p, q)
    rho, wa, wb = tutorial_on_cca(a, b)

    l = []
    for i in range(m):
        current_l = -(n - i - 0.5 * (p + q + 1))
        current_l -= np.sum(np.power(rho[:i], -2))
        current_l *= np.sum(np.log(1 - np.square(rho[i:])))
        l.append(current_l)

    p_vals = []
    dofs = []
    for j in range(m):
        dof = (p - j) * (m - j)
        dofs.append(dof)
        current_chi = ss.chi2(dof)
        p_vals.append(1 - current_chi.cdf(l[j]))

    return rho, wa, wb, p_vals


def get_positives(a):
    return a[a > 0]
