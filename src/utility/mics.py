import secrets
import string
from pathlib import Path


def generate_random_filename() -> str:
    """Generate a random filename with 5 alphanumeric characters."""
    characters = string.ascii_letters + string.digits
    random_str = ''.join(secrets.choice(characters) for _ in range(5))
    return random_str


def get_project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).parent.parent.parent