import numpy as np

def _get_probabilities(frequencies, estimator='ml'):
    if estimator == 'ml':
        probabilities = _get_probabilities_ml(frequencies)
    else:
        # Throw exception
        pass
    return probabilities

def _get_probabilities_ml(frequencies):
    return frequencies / frequencies.sum()

def _information_entropy_formula(probabilities, base=2):
    log = np.log(probabilities)
    log[np.isinf(log)] = 0
    entropy = - np.sum(probabilities * log)
    return entropy / np.log(base)

def information_entropy(x, estimator='ml', base=2):
    """
    Information Entropy:

    $$ H(X) = - \\sum_{x \\in X} p(x) \\log p(x) $$
    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each column of `x` represents a variable, and each row a single
        observation of all those variables.
    estimator: string, optional
        `ml`: maximum likelihood estimator
    base: float, optional

    Returns
    -------
    H : float
        Information Entropy of the variable or variables in x.
    """
    x = np.asarray(x)

    _, counts = _unique_rows(x, return_counts=True)
    frequencies = counts / counts.sum()
    entropy = _information_entropy_formula(frequencies, base=base)
    return entropy

def mutual_information(x, y=None, estimator='ml', base=2):
    """
    Mutual Information:

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each column of `x` represents a variable, and each row a single
        observation of all those variables.
    y :
    estimator: string, optional
        `ml`: maximum likelihood estimator
    base: float, optional

    Returns
    -------
    MI : float
    """
    x = np.asarray(x)
    if y is not None:
        y = np.asarray(y)
        xy = np.column_stack((x, y))
    else:
        xy = x
    entropy_x = information_entropy(xy[:, 0], estimator=estimator, base=base)
    entropy_y = information_entropy(xy[:, 1], estimator=estimator, base=base)
    entropy_xy = information_entropy(xy, estimator=estimator, base=base)
    mi = entropy_x + entropy_y - entropy_xy
    return mi

def condititional_entropy(x, y=None, estimator='ml', base=2):
    """
    Conditiona Entropy

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each column of `x` represents a variable, and each row a single
        observation of all those variables.
    y :
    estimator: string, optional
        `ml`: maximum likelihood estimator
    base: float, optional

    Returns
    -------
    CE : float
    """
    if y is not None:
        xy = np.column_stack((x, y))
    else:
        xy = x
    entropy_y = information_entropy(xy[:, 1], estimator=estimator, base=base)
    entropy_xy = information_entropy(xy, estimator=estimator, base=base)
    ce = entropy_xy - entropy_y
    return ce

def conditional_mutual_information(x, y=None, z=None, estimator='ml', base=2):
    """
    Conditional Mutual Information:

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each column of `x` represents a variable, and each row a single
        observation of all those variables.
    y : array_like, optional
    z : array_like, optional
    estimator: string, optional
        `ml`: maximum likelihood estimator
    base: float, optional

    Returns
    -------
    CMI : float
    """
    if y is not None and z is not None:
        xyz = np.column_stack((x, y, z))
    elif y is not None and z is None and x.shape[1] == 2:
        xyz = np.column_stack((x, y))
    else:
        xyz = x
    entropy_xz = information_entropy(xyz[:, (0, 2)], estimator=estimator, base=base)
    entropy_yz = information_entropy(xyz[:, (1, 2)], estimator=estimator, base=base)
    entropy_xyz = information_entropy(xyz, estimator=estimator, base=base)
    entropy_z = information_entropy(xyz[:, 2], estimator=estimator, base=base)
    cmi = _conditional_mutual_information_formula(entropy_xz, entropy_yz,
                                                  entropy_xyz, entropy_z)
    return cmi

def interaction_information(x, y=None, z=None, estimator='ml', base=2):
    """
    Interaction Information:

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each column of `x` represents a variable, and each row a single
        observation of all those variables.
    y : array_like, optional
    z : array_like, optional
    estimator: string, optional
        `ml`: maximum likelihood estimator
    base: float, optional

    Returns
    -------
    II : float
    """
    xyz = get_xyz(x,y,z)
    cmi = conditional_mutual_information(xyz, estimator=estimator, base=base)
    mi_xy = mutual_information(xyz[:,:2], estimator=estimator, base=base)
    ii = _interaction_information_formula(cmi, mi_xy)
    return ii

def partial_information_decomposition(x, y=None, z=None, estimator = 'ml', base = 2):
    xyz = get_xyz(x, y, z)
    frequencies = get_frequencies(xyz)
    # probabilities
    probabilities_xyz = frequencies / frequencies.sum()
    probabilities_yz = probabilities_xyz.sum(0)
    probabilities_xz = probabilities_xyz.sum(1)
    probabilities_xy = probabilities_xyz.sum(2)
    probabilities_z = probabilities_yz.sum(0)
    probabilities_y = probabilities_yz.sum(1)
    probabilities_x = probabilities_xz.sum(1)
    # For unique and synergy
    entropy_x = _information_entropy_formula(probabilities_x, base=base)
    entropy_y = _information_entropy_formula(probabilities_y, base=base)
    entropy_z = _information_entropy_formula(probabilities_z, base=base)
    entropy_xy = _information_entropy_formula(probabilities_xy, base=base)
    entropy_xz = _information_entropy_formula(probabilities_xz, base=base)
    entropy_yz = _information_entropy_formula(probabilities_yz, base=base)
    mi_xy = _mutual_information_formula(entropy_x, entropy_y, entropy_xy)
    # unique
    mi_xz = _mutual_information_formula(entropy_x, entropy_z, entropy_xz)
    mi_yz = _mutual_information_formula(entropy_y, entropy_z, entropy_yz)
    # synergy
    entropy_xyz = _information_entropy_formula(probabilities_xyz, base=base)
    cmi = _conditional_mutual_information_formula(entropy_xz, entropy_yz,
                                                  entropy_xyz, entropy_z)
    ii = _interaction_information_formula(cmi, mi_xy)
    redundancy = _redundancy(probabilities_z, probabilities_x, probabilities_y,
                             probabilities_xz, probabilities_yz)
    pid = dict()
    pid["redundancy"] = redundancy
    unique_x = mi_xz - redundancy
    unique_y = mi_yz - redundancy
    # Rounding errors may lead to slightly negative results
    pid["unique_1"] = np.maximum(0, unique_x)
    pid["unique_2"] = np.maximum(0, unique_y)
    synergy = ii + redundancy
    # Rounding errors may lead to slightly negative results
    pid["synergy"] = np.maximum(0, synergy)
    return pid

def _mutual_information_formula(entropy_x, entropy_y, entropy_xy):
    return entropy_x + entropy_y - entropy_xy

def _conditional_entropy_formula(entropy_xy, entropy_y):
    return entropy_xy - entropy_y

def _conditional_mutual_information_formula(entropy_xz, entropy_yz, entropy_xyz, entropy_z):
    return entropy_xz + entropy_yz - entropy_xyz - entropy_z

def _interaction_information_formula(conditional_mutual_information, mutual_information):
    return conditional_mutual_information - mutual_information

def _specific_information_formula(probabilities_source_target_i,
                                  probabilities_source,
                                  probabilities_target_i,
                                  base=2):
    log = np.log(probabilities_source_target_i / (probabilities_source * probabilities_target_i))
    log[np.isinf(log)] = 0
    specific_information = np.sum((probabilities_source_target_i / probabilities_target_i)
                                   * log)
    return specific_information / np.log(base)

def _redundancy(probabilities_target,
                probabilities_source_1,
                probabilities_source_2,
                probabilities_source_1_target,
                probabilities_source_2_target,
                target_dimension=None):
    number_of_target_states = len(probabilities_target)
    minimum_specific_information = np.zeros(number_of_target_states)
    for (i, probability_target) in enumerate(probabilities_target):
        specific_information_source_1 = _specific_information_formula(probabilities_source_1_target[:,i],
                                                                 probabilities_source_1,
                                                                 probability_target)
        specific_information_source_2 = _specific_information_formula(probabilities_source_2_target[:,i],
                                                                 probabilities_source_2,
                                                                 probability_target)
        minimum_specific_information[i] = np.minimum(specific_information_source_1, specific_information_source_2)
    red = np.dot(probabilities_target,minimum_specific_information)
    return red

def get_xyz(x, y, z):
    if y is not None and z is not None:
        xyz = np.column_stack((x, y, z))
    elif y is not None and z is None and x.shape[1] == 2:
        xyz = np.column_stack((x, y))
    else:
        xyz = x
    return xyz

def get_frequencies(x):
    values, counts = _unique_rows(x, return_counts=True)
    if x.ndim == 1:
        frequencies = counts
    else:
        ndim = x.shape[1]
        symbols = [{v:j for j,v in enumerate(np.unique(x[:,i]))} for i in range(ndim)]
        shape = [len(s) for s in symbols]
        frequencies = np.zeros(shape)
        for value, count in zip(values,counts):
            idx = tuple(symbols[i][v] for i, v in enumerate(value))
            frequencies[idx] = count
    frequencies = frequencies / frequencies.sum()
    return frequencies

def _unique_rows(x, return_counts=False):
    if x.ndim == 1:
        return np.unique(x, return_counts=return_counts)
    else:
        dtype = np.dtype((np.void, x.dtype.itemsize * x.shape[1]))
        y = np.ascontiguousarray(x).view(dtype)
        _, idx, counts = np.unique(y, return_index=True, return_counts=True)
        unique_x = x[idx,:]
        if return_counts:
            return unique_x, counts
        else:
            return unique_x
