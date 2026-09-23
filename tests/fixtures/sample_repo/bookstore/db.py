"""Database access for the bookstore."""

import functools
import os
import sqlite3

DB_PATH = os.environ.get("BOOKSTORE_DB", "bookstore.sqlite3")


@functools.lru_cache(maxsize=1)
def get_connection():
    """Open the SQLite connection once and reuse it."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


class BookRepository:
    """Reads and writes books."""

    def __init__(self, conn=None):
        self.conn = conn or get_connection()

    def add_book(self, title, author):
        self.conn.execute(
            "INSERT INTO books (title, author) VALUES (?, ?)", (title, author)
        )
        self.conn.commit()

    def find_by_author(self, author):
        rows = self.conn.execute("SELECT * FROM books WHERE author = ?", (author,))
        return [dict(row) for row in rows]


if __name__ == "__main__":
    print(BookRepository().find_by_author("Tolkien"))
