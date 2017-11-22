def probabilities(frequencies, estimator='ml'):
    if estimator == 'ml':
        probabilities = _get_probabilities_ml(frequencies)
    else:
        # Throw exception
        pass
    return probabilities


def _get_probabilities_ml(frequencies):
    return frequencies / frequencies.sum()