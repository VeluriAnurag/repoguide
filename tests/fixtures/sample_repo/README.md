# Bookstore

A tiny example application used as a test fixture for RepoGuide.

## Setup

Install dependencies and set `BOOKSTORE_DB` to choose the database file.

```bash
# This is a comment inside a code block, not a heading
pip install -r requirements.txt
```

## Architecture

- `bookstore/db.py` opens the SQLite connection and stores books.
- `bookstore/auth.py` hashes passwords and authenticates users.
