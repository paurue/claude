import numpy as np


def information_entropy(probabilities, base=2):
    log = np.log(probabilities)
    log[np.isinf(log)] = 0
    entropy = - np.sum(probabilities * log)
    return entropy / np.log(base)


def mutual_information(entropy_x, entropy_y, entropy_xy):
    return entropy_x + entropy_y - entropy_xy


def conditional_entropy(entropy_xy, entropy_y):
    return entropy_xy - entropy_y


def conditional_mutual_information(entropy_xz, entropy_yz,
                                   entropy_xyz, entropy_z):
    return entropy_xz + entropy_yz - entropy_xyz - entropy_z


def interaction_information(conditional_mutual_information,
                            mutual_information):
    return conditional_mutual_information - mutual_information


def specific_information(probabilities_source_target_i,
                         probabilities_source,
                         probabilities_target_i,
                         base=2):
    log = np.log(probabilities_source_target_i /
                 (probabilities_source * probabilities_target_i))
    log[np.isinf(log)] = 0
    specific_information = np.sum((probabilities_source_target_i /
                                   probabilities_target_i) * log)
    return specific_information / np.log(base)


def redundancy(probabilities_target,
               probabilities_source_1,
               probabilities_source_2,
               probabilities_source_1_target,
               probabilities_source_2_target,
               target_dimension=None):
    number_of_target_states = len(probabilities_target)
    minimum_specific_information = np.zeros(number_of_target_states)
    for (i, probability_target) in enumerate(probabilities_target):
        specific_information_source_1 = \
            specific_information(probabilities_source_1_target[:, i],
                                 probabilities_source_1,
                                 probability_target)
        specific_information_source_2 = \
            specific_information(probabilities_source_2_target[:, i],
                                 probabilities_source_2,
                                 probability_target)
        minimum_specific_information[i] = \
            np.minimum(specific_information_source_1,
                       specific_information_source_2)
    red = np.dot(probabilities_target, minimum_specific_information)
    return red
