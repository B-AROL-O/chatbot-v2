"""Create (or recreate) the SQLite database from schema.sql."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "cdo.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA_PATH.read_text())
    con.commit()
    con.close()
    print(f"Database initialised at {db_path}")


if __name__ == "__main__":
    init_db()
