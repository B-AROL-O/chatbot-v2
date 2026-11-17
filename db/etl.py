"""ETL: load the three CDO Excel files into the SQLite database.

Run order:
  python db/init_db.py   # create empty schema
  python db/etl.py       # populate all tables
"""

import sqlite3
from functools import partial
from pathlib import Path

import openpyxl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
INPUT_DIR = ROOT / "data" / "input" / "CDO"
DB_PATH = ROOT / "data" / "cdo.db"

CDO_REF_FILE = INPUT_DIR / "CDO reference.xlsx"
CONTENITORI_FILE = INPUT_DIR / "DataExport CONTENITORI.xlsx"
TAPPI_FILE = INPUT_DIR / "DataExport TAPPI.xlsx"

# ---------------------------------------------------------------------------
# Known machine-type labels (Tipo code -> Italian full name)
# ---------------------------------------------------------------------------

TIPO_LABELS = {
    "AL": "Accessori / Sistema di Alimentazione",
    "EL": "Elevatore",
    "M":  "Macchina Automatica / Semiautomatica",
    "MB": "Monoblocco",
    "SC": "Sistema di Controllo",
    "TS": "Torretta Automatica Rotativa Superiore",
    "TSI": "Torretta Automatica Rotativa Superiore e Inferiore",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(value) -> str | None:
    """Return stripped string or None for blank/dash values."""
    if value is None:
        return None
    s = str(value).strip()
    return None if s in ("", "-", "X") else s


def _row_real(row: dict, col: str) -> float | None:
    """Read `col` off `row` as a float, tolerating comma decimals and junk."""
    v = row.get(col)
    try:
        return float(str(v).replace(",", ".")) if v is not None else None
    except (ValueError, TypeError):
        return None


def _row_int(row: dict, col: str) -> int | None:
    """Read `col` off `row` as an int, or None if absent/unparseable."""
    v = row.get(col)
    try:
        return int(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def _load_sheet(path: Path) -> tuple[list[str], list[dict]]:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]
    # openpyxl pads every row out to ws.max_column, so headers and row always
    # agree in length; stay non-strict so a hand-edited sheet degrades to
    # dropped columns rather than aborting the whole load.
    rows = [dict(zip(headers, row, strict=False)) for row in ws.iter_rows(min_row=2, values_only=True)]
    return headers, rows


def _column_by_prefix(headers: list, prefix: str) -> str:
    """The one header that starts with ``prefix``; raise unless exactly one does."""
    matches = [h for h in headers if isinstance(h, str) and h.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one column starting with {prefix!r}, found {matches}")
    return matches[0]


def _get_or_insert(cur: sqlite3.Cursor, table: str, name_col: str,
                   id_col: str, value: str) -> int | None:
    """Return the id for *value* in a simple lookup table, inserting if needed."""
    if value is None:
        return None
    cur.execute(f"SELECT {id_col} FROM {table} WHERE {name_col} = ?", (value,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(f"INSERT INTO {table}({name_col}) VALUES (?)", (value,))
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Phase 1 – populate dim_machine_type
# ---------------------------------------------------------------------------

def load_machine_types(cur: sqlite3.Cursor) -> None:
    for code, label in TIPO_LABELS.items():
        cur.execute(
            "INSERT OR IGNORE INTO dim_machine_type(tipo_code, tipo_label) VALUES (?,?)",
            (code, label),
        )


# ---------------------------------------------------------------------------
# Phase 2 – populate dim_model and machine from CDO reference
# ---------------------------------------------------------------------------

def load_machines(cur: sqlite3.Cursor, rows: list[dict]) -> None:
    seen: set[str] = set()
    for r in rows:
        mat = _clean(r.get("Matricola"))
        if mat is None or mat in seen:
            continue
        seen.add(mat)

        tipo = _clean(r.get("Tipo"))
        model_name = _clean(r.get("Modello prodotto principale"))
        nt_raw = r.get("NT")
        nt = int(nt_raw) if nt_raw is not None else None
        diam = _clean(r.get("Diam Prim"))
        rot = _clean(r.get("Rotazione"))

        # dim_model
        model_id = None
        if model_name:
            cur.execute(
                "INSERT OR IGNORE INTO dim_model(model_name, tipo_code) VALUES (?,?)",
                (model_name, tipo),
            )
            cur.execute("SELECT model_id FROM dim_model WHERE model_name=?", (model_name,))
            model_id = cur.fetchone()[0]

        cur.execute(
            """INSERT OR IGNORE INTO machine(matricola, tipo_code, model_id, nt, diam_prim, rotazione)
               VALUES (?,?,?,?,?,?)""",
            (mat, tipo, model_id, nt, diam, rot),
        )


# ---------------------------------------------------------------------------
# Phase 3 – populate order-related dims and the order / order_sector tables
# ---------------------------------------------------------------------------

def load_orders(cur: sqlite3.Cursor, rows: list[dict], buyer_ref_col: str) -> None:
    for r in rows:
        cdo = _clean(r.get("CDO esportata"))
        if cdo is None:
            continue

        # dim_product_category
        raw_cat = str(r.get("Tipologia prodotto venduto") or "").strip()
        is_cancelled = 1 if "ANNULLATO" in raw_cat.upper() else 0
        cat_name = raw_cat.replace("\nORDINE ANNULLATO", "").replace("\nORDINE SOSPESO", "").strip()
        category_id = None
        if cat_name:
            cur.execute(
                "INSERT OR IGNORE INTO dim_product_category(category_name, is_cancelled) VALUES (?,?)",
                (cat_name, is_cancelled),
            )
            cur.execute("SELECT category_id FROM dim_product_category WHERE category_name=?", (cat_name,))
            category_id = cur.fetchone()[0]

        customer_id = _get_or_insert(cur, "dim_customer", "customer_name", "customer_id",
                                     _clean(r.get("Cliente")))
        buyer_ref_id = _get_or_insert(cur, "dim_buyer_ref", "buyer_ref_name", "buyer_ref_id",
                                      _clean(r.get(buyer_ref_col)))
        end_user_id = _get_or_insert(cur, "dim_end_user", "end_user_name", "end_user_id",
                                     _clean(r.get("Cliente Utilizzatore Finale")))

        # dim_country
        country_name = _clean(r.get("Stato/State"))
        continent = _clean(r.get("Continente"))
        country_id = None
        if country_name:
            cur.execute(
                "INSERT OR IGNORE INTO dim_country(country_name, continent) VALUES (?,?)",
                (country_name, continent),
            )
            cur.execute("SELECT country_id FROM dim_country WHERE country_name=?", (country_name,))
            country_id = cur.fetchone()[0]

        mat = _clean(r.get("Matricola"))
        cdo_principale = _clean(r.get("CDO prodotto principale"))
        rev_ord = _clean(r.get("Rev Ord"))
        rev_off_raw = r.get("Rev Off")
        rev_off = int(rev_off_raw) if rev_off_raw is not None else None
        anno_raw = r.get("Anno")
        anno = int(anno_raw) if anno_raw is not None else None
        prodotto = _clean(r.get("Prodotto"))

        cur.execute(
            """INSERT OR IGNORE INTO "order"(
                cdo_esportata, matricola, cdo_prodotto_principale, category_id,
                rev_ord, rev_off, anno, customer_id, buyer_ref_id, end_user_id,
                country_id, prodotto)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (cdo, mat, cdo_principale, category_id,
             rev_ord, rev_off, anno, customer_id, buyer_ref_id, end_user_id,
             country_id, prodotto),
        )

        # order_sector junction
        raw_settore = r.get("Settore merceologico") or ""
        for part in str(raw_settore).split(","):
            sector_name = part.strip()
            if not sector_name:
                continue
            sector_id = _get_or_insert(cur, "dim_sector", "sector_name", "sector_id", sector_name)
            cur.execute(
                "INSERT OR IGNORE INTO order_sector(cdo_esportata, sector_id) VALUES (?,?)",
                (cdo, sector_id),
            )


# ---------------------------------------------------------------------------
# Phase 4 – container samples
# ---------------------------------------------------------------------------

def load_containers(cur: sqlite3.Cursor, rows: list[dict]) -> None:
    for r in rows:
        codice = _clean(r.get("Codice campione contenitore"))
        if codice is None:
            continue

        ct_id = _get_or_insert(cur, "dim_container_type", "type_name", "container_type_id",
                               _clean(r.get("Tipo contenitore")))
        mat_id = _get_or_insert(cur, "dim_container_material", "material_name", "material_id",
                                _clean(r.get("Mat. contenitore")))

        _real = partial(_row_real, r)

        cur.execute(
            """INSERT OR IGNORE INTO container_sample(
                codice_campione, stato, matricola, cdo_prodotto_principale,
                desc_campione, container_type_id, forma_pianta, material_id,
                capacita, colore, costruttore, cod_costruttore,
                diam_collo, h_sotto_baga, diam_collo_h33, diam_collo_h43, diam_collo_h53,
                diam_corpo, diam_massimo, h_bottiglia, larghezza, lunghezza,
                bocchello_decentrato, note, nota_criticita, campione_critico)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                codice,
                _clean(r.get("Stato")),
                _clean(r.get("Matricola")),
                _clean(r.get("CDO prodotto principale")),
                _clean(r.get("Desc. Campione contenitore")),
                ct_id,
                _clean(r.get("Forma pianta contenitore")),
                mat_id,
                _clean(r.get("Capacità")),
                _clean(r.get("Colore")),
                _clean(r.get("Costruttore contenitore")),
                _clean(r.get("Cod. cont. del costr.")),
                _real("Diam. collo"),
                _real("H. sotto baga"),
                _real("Diam. collo H33"),
                _real("Diam. collo H43"),
                _real("Diam. collo H53"),
                _real("Diam. corpo contenitore"),
                _real("Diam. massimo"),
                _real("H. bott."),
                _real("Larghezza"),
                _real("Lunghezza"),
                _clean(r.get("Bocchello decentrato")),
                _clean(r.get("Note")),
                _clean(r.get("Nota criticità")),
                _clean(r.get("Campione critico")),
            ),
        )


# ---------------------------------------------------------------------------
# Phase 5 – cap samples
# ---------------------------------------------------------------------------

def load_caps(cur: sqlite3.Cursor, rows: list[dict]) -> None:
    for r in rows:
        codice = _clean(r.get("Codice campione tappo"))
        if codice is None:
            continue

        cap_type_id = _get_or_insert(cur, "dim_cap_type", "type_name", "cap_type_id",
                                     _clean(r.get("Sigla caps")))
        cap_mat_id = _get_or_insert(cur, "dim_cap_material", "material_name", "cap_material_id",
                                    _clean(r.get("Materiale caps.")))

        _real = partial(_row_real, r)
        _int = partial(_row_int, r)

        cur.execute(
            """INSERT OR IGNORE INTO cap_sample(
                codice_campione, stato, matricola, cdo_prodotto_principale,
                desc_campione, cap_type_id, cap_material_id, colore,
                costruttore, codice_costruttore,
                h_caps, h_pescante, h_senza_sigillo, h_totale,
                diam_massimo, diam_gambo, diam_testa, ingombri, neck_finish,
                peso_g, diam_min_dentatura, diam_max_dentatura,
                n_denti, n_principi_caps_1,
                sport_senza_appoggio, liner_interno,
                note, nota_criticita, campione_critico)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                codice,
                _clean(r.get("Stato")),
                _clean(r.get("Matricola")),
                _clean(r.get("CDO prodotto principale")),
                _clean(r.get("Desc. Campione tappo")),
                cap_type_id,
                cap_mat_id,
                _clean(r.get("Colore")),
                _clean(r.get("Costruttore caps.")),
                _clean(r.get("Codice caps. costruttore")),
                _real("H. caps."),
                _real("Altezza del pescante"),
                _real("H. senza sigillo"),
                _real("H. totale caps."),
                _real("Diam. massimo"),
                _real("Diam. gambo"),
                _real("Diam. testa"),
                _clean(r.get("Ingombri")),
                _clean(r.get("Neck finish")),
                _real("Peso [g]"),
                _real("Diam. min. dentatura"),
                _real("Diam. max. dentatura"),
                _int("N° denti"),
                _int("N° principi caps 1"),
                _clean(r.get("Sport senza appoggio")),
                _clean(r.get("Liner interno")),
                _clean(r.get("Note")),
                _clean(r.get("Nota criticità")),
                _clean(r.get("Campione critico")),
            ),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_etl(db_path: Path = DB_PATH) -> None:
    print("Loading Excel files …")
    cdo_headers, cdo_rows = _load_sheet(CDO_REF_FILE)
    # The header names the manufacturer ("Ref. acquirente da <company>").
    buyer_ref_col = _column_by_prefix(cdo_headers, "Ref. acquirente da")
    _, cont_rows = _load_sheet(CONTENITORI_FILE)
    _, tappi_rows = _load_sheet(TAPPI_FILE)
    print(f"  CDO reference:      {len(cdo_rows)} rows")
    print(f"  Contenitori:        {len(cont_rows)} rows")
    print(f"  Tappi:              {len(tappi_rows)} rows")

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    cur = con.cursor()

    print("\nPhase 1 – machine types …")
    load_machine_types(cur)

    print("Phase 2 – machines & models …")
    load_machines(cur, cdo_rows)

    print("Phase 3 – orders & sectors …")
    load_orders(cur, cdo_rows, buyer_ref_col)

    print("Phase 4 – container samples …")
    load_containers(cur, cont_rows)

    print("Phase 5 – cap samples …")
    load_caps(cur, tappi_rows)

    con.commit()
    con.close()

    # Verification counts
    print("\nRow counts:")
    con = sqlite3.connect(db_path)
    for table in [
        "dim_machine_type", "dim_model", "dim_product_category",
        "dim_sector", "dim_customer", "dim_end_user", "dim_buyer_ref", "dim_country",
        "dim_container_type", "dim_container_material", "dim_cap_type", "dim_cap_material",
        "machine", '"order"', "order_sector", "container_sample", "cap_sample",
    ]:
        (count,) = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"  {table.strip('\"'):30s} {count:>6}")
    con.close()


if __name__ == "__main__":
    run_etl()
