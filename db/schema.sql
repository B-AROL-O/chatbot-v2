-- CDO Reference normalized schema (SQLite / SQL-standard)
-- Three source files share Matricola + CDO prodotto principale as join keys:
--   CDO reference.xlsx        (780 rows)  orders / machines sold
--   DataExport CONTENITORI.xlsx (2045 rows) container/bottle samples
--   DataExport TAPPI.xlsx       (1714 rows) cap/closure samples

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Dimension / lookup tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dim_machine_type (
    tipo_code   TEXT PRIMARY KEY,
    tipo_label  TEXT
);

CREATE TABLE IF NOT EXISTS dim_model (
    model_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name  TEXT NOT NULL UNIQUE,
    tipo_code   TEXT REFERENCES dim_machine_type(tipo_code)
);

CREATE TABLE IF NOT EXISTS dim_product_category (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE,
    is_cancelled  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS dim_sector (
    sector_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    sector_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_customer (
    customer_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_end_user (
    end_user_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    end_user_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_buyer_ref (
    buyer_ref_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    buyer_ref_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_country (
    country_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    country_name  TEXT NOT NULL UNIQUE,
    continent     TEXT
);

CREATE TABLE IF NOT EXISTS dim_container_type (
    container_type_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    type_name          TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_container_material (
    material_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    material_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_cap_type (
    cap_type_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    type_name     TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_cap_material (
    cap_material_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    material_name    TEXT NOT NULL UNIQUE
);

-- ---------------------------------------------------------------------------
-- Core entity tables
-- ---------------------------------------------------------------------------

-- Physical machines (Matricola is the stable product identity)
CREATE TABLE IF NOT EXISTS machine (
    matricola   TEXT PRIMARY KEY,
    tipo_code   TEXT REFERENCES dim_machine_type(tipo_code),
    model_id    INTEGER REFERENCES dim_model(model_id),
    nt          INTEGER,
    diam_prim   TEXT,
    rotazione   TEXT
);

-- Orders / CDOs
CREATE TABLE IF NOT EXISTS "order" (
    cdo_esportata           TEXT PRIMARY KEY,
    matricola               TEXT REFERENCES machine(matricola),
    cdo_prodotto_principale TEXT,
    category_id             INTEGER REFERENCES dim_product_category(category_id),
    rev_ord                 TEXT,
    rev_off                 INTEGER,
    anno                    INTEGER,
    customer_id             INTEGER REFERENCES dim_customer(customer_id),
    buyer_ref_id            INTEGER REFERENCES dim_buyer_ref(buyer_ref_id),
    end_user_id             INTEGER REFERENCES dim_end_user(end_user_id),
    country_id              INTEGER REFERENCES dim_country(country_id),
    prodotto                TEXT
);

-- Junction: orders <-> sectors  (M:M)
CREATE TABLE IF NOT EXISTS order_sector (
    cdo_esportata  TEXT REFERENCES "order"(cdo_esportata),
    sector_id      INTEGER REFERENCES dim_sector(sector_id),
    PRIMARY KEY (cdo_esportata, sector_id)
);

-- Container / bottle samples
CREATE TABLE IF NOT EXISTS container_sample (
    codice_campione         TEXT PRIMARY KEY,
    stato                   TEXT,
    matricola               TEXT REFERENCES machine(matricola),
    cdo_prodotto_principale TEXT,
    desc_campione           TEXT,
    container_type_id       INTEGER REFERENCES dim_container_type(container_type_id),
    forma_pianta            TEXT,
    material_id             INTEGER REFERENCES dim_container_material(material_id),
    capacita                TEXT,
    colore                  TEXT,
    costruttore             TEXT,
    cod_costruttore         TEXT,
    diam_collo              REAL,
    h_sotto_baga            REAL,
    diam_collo_h33          REAL,
    diam_collo_h43          REAL,
    diam_collo_h53          REAL,
    diam_corpo              REAL,
    diam_massimo            REAL,
    h_bottiglia             REAL,
    larghezza               REAL,
    lunghezza               REAL,
    bocchello_decentrato    TEXT,
    note                    TEXT,
    nota_criticita          TEXT,
    campione_critico        TEXT
);

-- Cap / closure samples
CREATE TABLE IF NOT EXISTS cap_sample (
    codice_campione         TEXT PRIMARY KEY,
    stato                   TEXT,
    matricola               TEXT REFERENCES machine(matricola),
    cdo_prodotto_principale TEXT,
    desc_campione           TEXT,
    cap_type_id             INTEGER REFERENCES dim_cap_type(cap_type_id),
    cap_material_id         INTEGER REFERENCES dim_cap_material(cap_material_id),
    colore                  TEXT,
    costruttore             TEXT,
    codice_costruttore      TEXT,
    h_caps                  REAL,
    h_pescante              REAL,
    h_senza_sigillo         REAL,
    h_totale                REAL,
    diam_massimo            REAL,
    diam_gambo              REAL,
    diam_testa              REAL,
    ingombri                TEXT,
    neck_finish             TEXT,
    peso_g                  REAL,
    diam_min_dentatura      REAL,
    diam_max_dentatura      REAL,
    n_denti                 INTEGER,
    n_principi_caps_1       INTEGER,
    sport_senza_appoggio    TEXT,
    liner_interno           TEXT,
    note                    TEXT,
    nota_criticita          TEXT,
    campione_critico        TEXT
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_order_matricola  ON "order"(matricola);
CREATE INDEX IF NOT EXISTS idx_order_customer   ON "order"(customer_id);
CREATE INDEX IF NOT EXISTS idx_order_country    ON "order"(country_id);
CREATE INDEX IF NOT EXISTS idx_order_anno       ON "order"(anno);
CREATE INDEX IF NOT EXISTS idx_container_mat    ON container_sample(matricola);
CREATE INDEX IF NOT EXISTS idx_cap_mat          ON cap_sample(matricola);
CREATE INDEX IF NOT EXISTS idx_order_sector_sec ON order_sector(sector_id);
