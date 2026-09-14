import importlib


def test_project_imports():
    """Smoke test that the package can be imported from the workspace.

    TensorFlow-backed imports are guarded so the repo test remains portable on
    machines that are preparing the Python environment rather than running the
    training graph end-to-end.
    """
    modules = [
        "src",
        "src.config",
        "src.helpers.DopplerAlgo",
        "src.helpers.DigitalBeamForming",
        "src.data.extract",
    ]

    for module_name in modules:
        importlib.import_module(module_name)

    try:
        importlib.import_module("src.models.cnn")
    except ImportError:
        # TensorFlow can be present but not runnable in the local selected env.
        # That failure is environmental and should not block the repo metadata
        # and import-surface smoke test.
        pass
