from .det_solver import DetSolver

__all__ = list(globals().keys())


def setup_environment():
    """setup function for preloading before rtdetrv2"""
    from rtdetrv2.solver import TASKS

    # NOTE: hacky way, replace the DetSolver with our implementation
    # TODO: support registeration/update API in rtdetrv2?
    TASKS["detection"] = DetSolver
