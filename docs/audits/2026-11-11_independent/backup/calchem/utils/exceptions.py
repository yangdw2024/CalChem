class CalChemError(Exception):
    pass


class InputError(CalChemError):
    pass


class CalculationError(CalChemError):
    pass


class ConvergenceError(CalculationError):
    pass


class BasisNotFoundError(InputError):
    pass


class MethodNotImplementedError(CalChemError):
    pass


class GPUNotAvailableError(CalChemError):
    pass
