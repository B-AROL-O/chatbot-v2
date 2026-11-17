"""Read-only SQL tool over the normalized CDO SQLite database (data/cdo.db)."""

import logging
import re
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Repo root: src/sme_chatbot/tools/cdo_sql.py -> tools -> sme_chatbot -> src -> root.
# (This module lived at tools/cdo_sql.py pre-WP2, one level shallower; the extra
# `.parent` below accounts for the src/sme_chatbot/ nesting introduced by the move.)
DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cdo.db"
MAX_ROWS = 200

# Schema summary fed to the LLM so it can write correct SQL.
SCHEMA_DESCRIPTION = f"""\
SQLite database of CDO orders, machines and packaging samples.
Three source files join on `matricola` and `cdo_prodotto_principale`.

Core tables:
  machine(matricola PK, tipo_code, model_id, nt, diam_prim, rotazione)
  "order"(cdo_esportata PK, matricola, cdo_prodotto_principale, category_id,
          rev_ord, rev_off, anno, customer_id, buyer_ref_id, end_user_id,
          country_id, prodotto)
  order_sector(cdo_esportata, sector_id)            -- M:M orders<->sectors
  container_sample(codice_campione PK, stato, matricola, cdo_prodotto_principale,
          desc_campione, container_type_id, forma_pianta, material_id, capacita,
          colore, costruttore, cod_costruttore, diam_collo, h_sotto_baga,
          diam_collo_h33, diam_collo_h43, diam_collo_h53, diam_corpo, diam_massimo,
          h_bottiglia, larghezza, lunghezza, bocchello_decentrato, note,
          nota_criticita, campione_critico)
  cap_sample(codice_campione PK, stato, matricola, cdo_prodotto_principale,
          desc_campione, cap_type_id, cap_material_id, colore, costruttore,
          codice_costruttore, h_caps, h_pescante, h_senza_sigillo, h_totale,
          diam_massimo, diam_gambo, diam_testa, ingombri, neck_finish, peso_g,
          diam_min_dentatura, diam_max_dentatura, n_denti, n_principi_caps_1,
          sport_senza_appoggio, liner_interno, note, nota_criticita, campione_critico)

Lookup tables (join on the *_id columns above):
  dim_machine_type(tipo_code PK, tipo_label)
  dim_model(model_id PK, model_name, tipo_code)
  dim_product_category(category_id PK, category_name, is_cancelled)
  dim_sector(sector_id PK, sector_name)
  dim_customer(customer_id PK, customer_name)
  dim_end_user(end_user_id PK, end_user_name)
  dim_buyer_ref(buyer_ref_id PK, buyer_ref_name)
  dim_country(country_id PK, country_name, continent)
  dim_container_type(container_type_id PK, type_name)
  dim_container_material(material_id PK, material_name)
  dim_cap_type(cap_type_id PK, type_name)
  dim_cap_material(cap_material_id PK, material_name)

Notes:
- `order` is a reserved word: always quote it as "order".
- Returns at most {MAX_ROWS} rows."""

# Only allow a single read-only SELECT (optionally a CTE starting with WITH).
_ALLOWED_PREFIX = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
# Block anything that could mutate or chain into a write.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|"
    r"pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)


def _is_safe(sql: str) -> bool:
    # Reject multiple statements (a stray semicolon could chain a write).
    if sql.strip().rstrip(";").count(";") > 0:
        return False
    if not _ALLOWED_PREFIX.match(sql):
        return False
    if _FORBIDDEN.search(sql):  # noqa: SIM103 - one rejection rule per line, deliberately
        return False
    return True


def query_cdo_db(sql: str) -> dict:
    """Run a read-only SELECT and return {"columns": [...], "rows": [...]}."""
    if not _is_safe(sql):
        return {"error": "Only a single read-only SELECT/WITH statement is allowed."}

    if not DB_PATH.exists():
        return {"error": f"Database not found at {DB_PATH}."}

    logger.info("query_cdo_db: %s", sql)
    try:
        # Read-only connection as a second line of defence.
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            cur = con.execute(sql)
            rows = cur.fetchmany(MAX_ROWS + 1)
            columns = [d[0] for d in cur.description] if cur.description else []
        finally:
            con.close()
    except sqlite3.Error as exc:
        logger.warning("SQL error: %s", exc)
        return {"error": f"SQL error: {exc}"}

    truncated = len(rows) > MAX_ROWS
    result = {
        "columns": columns,
        "rows": [dict(r) for r in rows[:MAX_ROWS]],
    }
    if truncated:
        result["note"] = f"Result truncated to {MAX_ROWS} rows."
    return result
