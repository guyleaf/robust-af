import pickle

_ENV_SETUP_DONE = False
# map old package/module to new namespace while unpicking objects
_MODULE_PREFIX_MAP = {
    "robust_au_od": "robust_af",
    "robust_af.models.robust_layers": "robust_af.models.feature_adapters",
}


class UnpicklerWrapper(pickle.Unpickler):
    def find_class(self, module_name: str, global_name: str):
        for k, v in _MODULE_PREFIX_MAP.items():
            if module_name.startswith(k):
                module_name = module_name.replace(k, v, 1)
                break
        return super().find_class(module_name, global_name)


def setup_environment():
    """Perform environment setup work for some backward compatibilities."""
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True

    pickle.Unpickler = UnpicklerWrapper
