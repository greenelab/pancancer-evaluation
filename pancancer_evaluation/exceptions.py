"""
Exceptions specific to pancancer evaluation experiments
"""

class NoTrainSamplesError(Exception):
    """
    Custom exception to raise when there are insufficient train samples for a
    given cancer type.

    This allows calling scripts to choose how to handle this case (e.g. to
    print an error message and continue, or to abort execution).
    """
    def __init__(self, *args):
        super().__init__(*args)


class NoTestSamplesError(Exception):
    """
    Custom exception to raise when there are insufficient test samples for a
    given cancer type.

    This allows calling scripts to choose how to handle this case (e.g. to
    print an error message and continue, or to abort execution).
    """
    def __init__(self, *args):
        super().__init__(*args)


class OneClassError(Exception):
    """
    Custom exception to raise when there is only one class present in the
    test set for the given cancer type.

    This allows calling scripts to choose how to handle this case (e.g. to
    print an error message and continue, or to abort execution).
    """
    def __init__(self, *args):
        super().__init__(*args)


class ResultsFileExistsError(Exception):
    """
    Custom exception to raise when the results file already exists for the
    given gene and cancer type.

    This allows calling scripts to choose how to handle this case (e.g. to
    print an error message and continue, or to abort execution).
    """
    def __init__(self, *args):
        super().__init__(*args)

