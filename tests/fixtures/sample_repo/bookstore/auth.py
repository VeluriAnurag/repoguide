"""User authentication."""

import hashlib
import hmac

SALT = b"bookstore-demo-salt"


def hash_password(password: str) -> str:
    return hashlib.sha256(SALT + password.encode()).hexdigest()


def authenticate_user(username: str, password: str, users: dict[str, str]) -> bool:
    """Return True if the password matches the stored hash for this user."""
    stored_hash = users.get(username)
    if stored_hash is None:
        return False
    return hmac.compare_digest(stored_hash, hash_password(password))
