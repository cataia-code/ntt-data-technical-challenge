"""Carga y transformación del dataset de consumo doméstico de café."""

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "coffee_db.parquet"
LONG_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "coffee_long.parquet"

# El nombre viene con mojibake en el parquet ("C�te d'Ivoire"); se corrige al cargar.
MOJIBAKE_FIX = {"C�te d'Ivoire": "Côte d'Ivoire"}

# Código ISO-3 por país para el mapa geográfico de Plotly (locationmode='ISO-3').
# Robusto ante nombres no estándar del ICO (ej. "Viet Nam", "Bolivia (Plurinational State of)").
COUNTRY_ISO3 = {
    "Angola": "AGO",
    "Bolivia (Plurinational State of)": "BOL",
    "Brazil": "BRA",
    "Burundi": "BDI",
    "Ecuador": "ECU",
    "Indonesia": "IDN",
    "Madagascar": "MDG",
    "Malawi": "MWI",
    "Papua New Guinea": "PNG",
    "Paraguay": "PRY",
    "Peru": "PER",
    "Rwanda": "RWA",
    "Timor-Leste": "TLS",
    "Zimbabwe": "ZWE",
    "Congo": "COG",
    "Cuba": "CUB",
    "Dominican Republic": "DOM",
    "Haiti": "HTI",
    "Philippines": "PHL",
    "Tanzania": "TZA",
    "Zambia": "ZMB",
    "Cameroon": "CMR",
    "Central African Republic": "CAF",
    "Colombia": "COL",
    "Costa Rica": "CRI",
    "Côte d'Ivoire": "CIV",
    "Democratic Republic of Congo": "COD",
    "El Salvador": "SLV",
    "Equatorial Guinea": "GNQ",
    "Ethiopia": "ETH",
    "Gabon": "GAB",
    "Ghana": "GHA",
    "Guatemala": "GTM",
    "Guinea": "GIN",
    "Guyana": "GUY",
    "Honduras": "HND",
    "India": "IND",
    "Jamaica": "JAM",
    "Kenya": "KEN",
    "Lao People's Democratic Republic": "LAO",
    "Liberia": "LBR",
    "Mexico": "MEX",
    "Nepal": "NPL",
    "Nicaragua": "NIC",
    "Nigeria": "NGA",
    "Panama": "PAN",
    "Sierra Leone": "SLE",
    "Sri Lanka": "LKA",
    "Thailand": "THA",
    "Togo": "TGO",
    "Trinidad & Tobago": "TTO",
    "Uganda": "UGA",
    "Venezuela": "VEN",
    "Viet Nam": "VNM",
    "Yemen": "YEM",
}

# Continente por país (agregación geográfica para el EDA). Asignación estándar de Naciones Unidas;
# Papua New Guinea es el único caso de Oceanía en el dataset (n=1, se reporta igual por transparencia).
COUNTRY_CONTINENT = {
    "Angola": "África", "Burundi": "África", "Madagascar": "África", "Malawi": "África",
    "Rwanda": "África", "Zimbabwe": "África", "Congo": "África", "Tanzania": "África",
    "Zambia": "África", "Cameroon": "África", "Central African Republic": "África",
    "Côte d'Ivoire": "África", "Democratic Republic of Congo": "África", "Equatorial Guinea": "África",
    "Ethiopia": "África", "Gabon": "África", "Ghana": "África", "Guinea": "África", "Kenya": "África",
    "Liberia": "África", "Nigeria": "África", "Sierra Leone": "África", "Togo": "África",
    "Uganda": "África",
    "Bolivia (Plurinational State of)": "América", "Brazil": "América", "Ecuador": "América",
    "Paraguay": "América", "Peru": "América", "Cuba": "América", "Dominican Republic": "América",
    "Haiti": "América", "Costa Rica": "América", "Colombia": "América", "El Salvador": "América",
    "Guatemala": "América", "Guyana": "América", "Honduras": "América", "Jamaica": "América",
    "Mexico": "América", "Nicaragua": "América", "Panama": "América", "Trinidad & Tobago": "América",
    "Venezuela": "América",
    "Indonesia": "Asia", "Timor-Leste": "Asia", "Philippines": "Asia", "India": "Asia",
    "Lao People's Democratic Republic": "Asia", "Nepal": "Asia", "Sri Lanka": "Asia",
    "Thailand": "Asia", "Viet Nam": "Asia", "Yemen": "Asia",
    "Papua New Guinea": "Oceanía",
}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["Country"] = df["Country"].replace(MOJIBAKE_FIX)
    return df


def validate_coherence(df: pd.DataFrame) -> dict:
    """Verifica que Total_domestic_consumption == suma de las columnas anuales (sin negativos).

    Devuelve un dict con el resultado; lanza AssertionError si la coherencia no se cumple.
    """
    year_cols = [c for c in df.columns if "/" in c]
    row_sum = df[year_cols].sum(axis=1)
    max_diff = int((df["Total_domestic_consumption"] - row_sum).abs().max())
    has_negatives = bool((df[year_cols] < 0).any().any())
    assert max_diff == 0, f"Incoherencia Total vs suma anual: max |diff| = {max_diff}"
    assert not has_negatives, "Existen valores de consumo negativos"
    return {"rows": len(df), "max_abs_diff": max_diff, "has_negatives": has_negatives}


def parse_crop_year(col: str) -> int:
    """"1990/91" -> 1990 (año fiscal de inicio, convención ICO)."""
    return int(col.split("/")[0])


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    year_cols = [c for c in df.columns if "/" in c]
    long_df = df.melt(
        id_vars=["Country", "Coffee type"],
        value_vars=year_cols,
        var_name="crop_year",
        value_name="consumption",
    )
    long_df["fiscal_year_start"] = long_df["crop_year"].apply(parse_crop_year)
    long_df["iso3"] = long_df["Country"].map(COUNTRY_ISO3)
    long_df["continent"] = long_df["Country"].map(COUNTRY_CONTINENT)
    long_df = long_df.sort_values(["Country", "fiscal_year_start"]).reset_index(drop=True)
    return long_df


def flag_zero_series(long_df: pd.DataFrame) -> pd.DataFrame:
    """Marca países cuya serie completa es cero (sin datos utilizables)."""
    totals = long_df.groupby("Country")["consumption"].sum()
    zero_countries = totals[totals == 0].index
    long_df = long_df.copy()
    long_df["is_valid_series"] = ~long_df["Country"].isin(zero_countries)
    return long_df


def build_long_dataset(raw_path: Path = RAW_PATH, save_to: Path = LONG_PATH) -> pd.DataFrame:
    df = load_raw(raw_path)
    validate_coherence(df)
    long_df = wide_to_long(df)
    long_df = flag_zero_series(long_df)
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_parquet(save_to, index=False)
    return long_df


if __name__ == "__main__":
    raw = load_raw()
    print("Coherencia:", validate_coherence(raw))
    out = build_long_dataset()
    missing_iso = out.loc[out["iso3"].isna(), "Country"].unique().tolist()
    missing_cont = out.loc[out["continent"].isna(), "Country"].unique().tolist()
    print(f"coffee_long.parquet: {out.shape}, sin ISO-3: {missing_iso}, sin continente: {missing_cont}, "
          f"inválidos: {out.loc[~out['is_valid_series'], 'Country'].unique().tolist()}")
