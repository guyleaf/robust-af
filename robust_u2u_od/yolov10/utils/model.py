
from huggingface_hub import repo_exists
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id


def is_huggingface_hub_model(model: str):
    try:
        validate_repo_id(model)
        return repo_exists(model)
    except HFValidationError:
        return False
