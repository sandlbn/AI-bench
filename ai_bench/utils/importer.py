import importlib.util
import os
import sys
import types


# From importlib docs:
# https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
def import_from_path(
    module_name: str, file_path: str | os.PathLike
) -> types.ModuleType:
    """Import a module directly from a source file.
    Args:
        module_name: Name of the module
        file_path: Absolute path to a Python file
    Returns:
        Loaded module
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
