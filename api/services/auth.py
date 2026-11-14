import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")


def verify_api_key(provided_key: str) -> bool:
    """
    Verify the provided API key against the expected API key.

    Args:
        provided_key (str): The API key provided in the request.

    Returns:
        bool: True if the provided key matches the expected key, False otherwise.
    """
    return provided_key == API_KEY