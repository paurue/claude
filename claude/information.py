import numpy as np

from claude.utils.arrayfuncs import unique_rows, glue_xyz, get_frequencies
import claude.utils.formulas as F

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

    _, counts = unique_rows(x, return_counts=True)
    frequencies = counts / counts.sum()
    entropy = F.information_entropy(frequencies, base=base)
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
    Conditional Entropy

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
    entropy_xz = information_entropy(xyz[:, (0, 2)], estimator=estimator,
                                     base=base)
    entropy_yz = information_entropy(xyz[:, (1, 2)], estimator=estimator,
                                     base=base)
    entropy_xyz = information_entropy(xyz, estimator=estimator,
                                      base=base)
    entropy_z = information_entropy(xyz[:, 2], estimator=estimator,
                                    base=base)
    cmi = F.conditional_mutual_information(entropy_xz, entropy_yz,
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
    xyz = glue_xyz(x, y, z)
    cmi = conditional_mutual_information(xyz, estimator=estimator, base=base)
    mi_xy = mutual_information(xyz[:, :2], estimator=estimator, base=base)
    ii = F.interaction_information(cmi, mi_xy)
    return ii


def partial_information_decomposition(x, y=None, z=None, estimator='ml',
                                      base=2):
    xyz = glue_xyz(x, y, z)
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
    entropy_x = F.information_entropy(probabilities_x, base=base)
    entropy_y = F.information_entropy(probabilities_y, base=base)
    entropy_z = F.information_entropy(probabilities_z, base=base)
    entropy_xy = F.information_entropy(probabilities_xy, base=base)
    entropy_xz = F.information_entropy(probabilities_xz, base=base)
    entropy_yz = F.information_entropy(probabilities_yz, base=base)
    mi_xy = F.mutual_information(entropy_x, entropy_y, entropy_xy)
    # unique
    mi_xz = F.mutual_information(entropy_x, entropy_z, entropy_xz)
    mi_yz = F.mutual_information(entropy_y, entropy_z, entropy_yz)
    # synergy
    entropy_xyz = F.information_entropy(probabilities_xyz, base=base)
    cmi = F.conditional_mutual_information(entropy_xz, entropy_yz,
                                           entropy_xyz, entropy_z)
    ii = F.interaction_information(cmi, mi_xy)
    redundancy = F.redundancy(probabilities_z, probabilities_x, probabilities_y,
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


