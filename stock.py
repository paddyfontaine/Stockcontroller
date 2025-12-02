import io
import locale
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Versuche deutsches Locale für Formatierung zu setzen (falls verfügbar)
try:
    locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")
except locale.Error:
    # Falls nicht verfügbar, arbeiten wir mit eigener Formatlogik
    pass

# Pflichtspalten, die in der CSV erwartet werden
REQUIRED_COLUMNS: List[str] = [
    "Produktkategorie",
    "Category",
    "Hauptwarengruppe",
    "Warengruppe",
    "Marke",
    "Filiale",
    "Kanal",
    "Datenreihe",
    "Datumsbereich und Plan",
    "Umsatz Menge",
    "Bestand Menge (Vortag)",
    "Offene Bestell. Menge",
]

# Standard-Dimensionen für die Gruppierung (kann der User später umstellen)
DEFAULT_GROUP_COLS: List[str] = [
    "Produktkategorie",
    "Category",
    "Hauptwarengruppe",
    "Warengruppe",
    "Marke",
]

# Reihenfolge für kaskadierende Filter
FILTER_COLS: List[str] = [
    "Produktkategorie",
    "Category",
    "Hauptwarengruppe",
    "Warengruppe",
    "Marke",
    "Filiale",
]


def clean_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Entfernt eventuelles UTF-8 BOM aus dem ersten Spaltennamen."""
    if len(df.columns) > 0:
        first_col = df.columns[0]
        if isinstance(first_col, str) and first_col.startswith("\ufeff"):
            new_cols = list(df.columns)
            new_cols[0] = first_col.replace("\ufeff", "")
            df.columns = new_cols
    return df


def detect_and_read_csv(uploaded_file) -> pd.DataFrame:
    """
    Liest die CSV mit automatischer Erkennung von Trennzeichen und Encoding.
    Nutzt eine robuste Variante für unterschiedliche Exporte.
    """
    raw_bytes = uploaded_file.read()
    text_buffer = io.StringIO(raw_bytes.decode("utf-8", errors="replace"))
    df = pd.read_csv(text_buffer, sep=None, engine="python")
    df = clean_bom_columns(df)
    return df


def to_numeric_de(series: pd.Series) -> pd.Series:
    """
    Konvertiert eine Spalte aus deutschem Zahlenformat (Tausenderpunkt, Komma)
    in einen numerischen Wert. Nicht konvertierbare Werte werden zu 0.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0)

    cleaned = (
        series.astype(str)
        .str.replace(".", "", regex=False)  # Tausenderpunkte entfernen
        .str.replace(",", ".", regex=False)  # Komma durch Punkt ersetzen
        .str.strip()
        .replace({"": np.nan})
    )
    numeric = pd.to_numeric(cleaned, errors="coerce").fillna(0)
    return numeric


def ensure_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Prüft, ob alle Pflichtspalten vorhanden sind."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing


def format_int_de(value: float) -> str:
    """Formatiert Ganzzahlen mit Tausendertrennzeichen im deutschen Stil."""
    try:
        v = int(round(value))
        return f"{v:,}".replace(",", ".")
    except Exception:
        return "-"


def format_percent_de(value: float, decimals: int = 1) -> str:
    """Formatiert Prozentwerte im deutschen Stil."""
    try:
        fmt = f"{{:,.{decimals}f}}".format(value)
        fmt = fmt.replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{fmt} %"
    except Exception:
        return "-"


@st.cache_data(show_spinner=False)
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Bereitet die Daten vor: Typen, Kanalbereinigung, numerische Konvertierung."""
    df = df.copy()

    # BOM entfernen falls vorhanden
    df = clean_bom_columns(df)

    # Pflichtspalten prüfen
    ok, missing = ensure_required_columns(df)
    if not ok:
        raise ValueError("Es fehlen Pflichtspalten: " + ", ".join(missing))

    # Kanal bereinigen: null/leer → Stationär
    df["Kanal"] = df["Kanal"].astype(str)
    df.loc[
        df["Kanal"].str.lower().isin(["nan", "null", "none", ""]),
        "Kanal",
    ] = "Stationär"

    # Numerische Spalten konvertieren
    numeric_cols = ["Umsatz Menge", "Bestand Menge (Vortag)", "Offene Bestell. Menge"]
    for col in numeric_cols:
        df[col] = to_numeric_de(df[col])

    return df


def filter_by_channel(df: pd.DataFrame, kanal_option: str) -> pd.DataFrame:
    """Filtert nach Kanal gemäß Auswahl."""
    if kanal_option == "Nur Stationär":
        return df[df["Kanal"] == "Stationär"].copy()
    # Stationär + Online
    return df[df["Kanal"].isin(["Stationär", "Online"])].copy()


def aggregate_data(df: pd.DataFrame, kanal_option: str, group_cols: List[str]) -> pd.DataFrame:
    """
    Erstellt die aggregierte Sicht je Kombination der group_cols
    inkl. hochgerechneter Bestände, erwarteter Verkäufe, Deckungsquote und Status.
    """
    df = filter_by_channel(df, kanal_option)

    # Ist- und Vergleichszeitraum trennen
    ist_df = df[df["Datenreihe"] == "Ist-Zeitraum"].copy()
    comp_df = df[df["Datenreihe"] == "Vergleichszeitraum"].copy()

    if ist_df.empty:
        return pd.DataFrame()

    # Falls keine Dimension gewählt wurde → alles zu einer Zeile zusammenfassen
    if not group_cols:
        ist_grouped = (
            ist_df[["Bestand Menge (Vortag)", "Offene Bestell. Menge", "Umsatz Menge"]]
            .sum()
            .to_frame()
            .T
        )
        if not comp_df.empty:
            comp_grouped = comp_df[["Umsatz Menge"]].sum().to_frame().T
        else:
            comp_grouped = pd.DataFrame([{"Umsatz Menge": 0}])
    else:
        ist_grouped = (
            ist_df.groupby(group_cols, dropna=False)[
                ["Bestand Menge (Vortag)", "Offene Bestell. Menge", "Umsatz Menge"]
            ]
            .sum()
        )

        if not comp_df.empty:
            comp_grouped = (
                comp_df.groupby(group_cols, dropna=False)[["Umsatz Menge"]]
                .sum()
            )
        else:
            comp_grouped = pd.DataFrame(columns=["Umsatz Menge"])

    ist_grouped = ist_grouped.rename(
        columns={
            "Bestand Menge (Vortag)": "Bestand_vortag",
            "Offene Bestell. Menge": "Offene_bestellungen",
            "Umsatz Menge": "Umsatz_ist",
        }
    )

    comp_grouped = comp_grouped.rename(columns={"Umsatz Menge": "erwartete_verkaeufe"})

    # Zusammenführen
    result = ist_grouped.join(comp_grouped, how="left")

    # Fehlende erwartete Verkäufe als 0 behandeln
    if "erwartete_verkaeufe" not in result.columns:
        result["erwartete_verkaeufe"] = 0
    else:
        result["erwartete_verkaeufe"] = result["erwartete_verkaeufe"].fillna(0)

    # Hochgerechneter Bestand
    result["hochgerechneter_bestand"] = (
        result["Bestand_vortag"] + result["Offene_bestellungen"] - result["Umsatz_ist"]
    )

    # Deckungsquote: bei erwartete_verkaeufe = 0 fachlich 0 %
    result["deckungsquote_pct"] = np.where(
        result["erwartete_verkaeufe"] > 0,
        (result["hochgerechneter_bestand"] / result["erwartete_verkaeufe"]) * 100.0,
        0.0,
    )

    # Ampel-Status
    conditions = [
        result["hochgerechneter_bestand"] <= 0,
        (result["hochgerechneter_bestand"] > 0)
        & (result["hochgerechneter_bestand"] < result["erwartete_verkaeufe"]),
        result["hochgerechneter_bestand"] >= result["erwartete_verkaeufe"],
    ]
    choices = ["Out-of-Stock", "kritisch", "ausreichend"]
    result["status"] = np.select(conditions, choices, default="ausreichend")

    # Index zurück als Spalten
    result = result.reset_index()

    return result


def compute_kpis(agg_df: pd.DataFrame) -> dict:
    """Berechnet die wichtigsten KPIs für das KPI-Panel."""
    if agg_df.empty:
        return {
            "sum_bestand": 0,
            "sum_erwartet": 0,
            "avg_deckungsquote": 0,
            "count_oos": 0,
            "count_kritisch": 0,
        }

    sum_bestand = float(agg_df["hochgerechneter_bestand"].sum())
    sum_erwartet = float(agg_df["erwartete_verkaeufe"].sum())
    avg_deckung = float(agg_df["deckungsquote_pct"].mean())
    count_oos = int((agg_df["status"] == "Out-of-Stock").sum())
    count_kritisch = int((agg_df["status"] == "kritisch").sum())

    return {
        "sum_bestand": sum_bestand,
        "sum_erwartet": sum_erwartet,
        "avg_deckungsquote": avg_deckung,
        "count_oos": count_oos,
        "count_kritisch": count_kritisch,
    }


def make_display_df(agg_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Erstellt eine Anzeige-DataFrame mit formatierten Zahlen."""
    if agg_df.empty:
        return agg_df

    display_df = agg_df.copy()

    display_df["Hochgerechneter Bestand"] = agg_df["hochgerechneter_bestand"].apply(
        format_int_de
    )
    display_df["Erwartete Verkäufe"] = agg_df["erwartete_verkaeufe"].apply(
        format_int_de
    )
    display_df["Deckungsquote"] = agg_df["deckungsquote_pct"].apply(
        lambda v: format_percent_de(v, decimals=1)
    )

    base_cols = [c for c in group_cols if c in display_df.columns]
    metric_cols = [
        "Hochgerechneter Bestand",
        "Erwartete Verkäufe",
        "Deckungsquote",
        "status",
    ]
    other_cols = [c for c in display_df.columns if c not in base_cols + metric_cols]
    ordered_cols = base_cols + metric_cols + other_cols

    display_df = display_df[ordered_cols].rename(columns={"status": "Status"})

    return display_df


def main() -> None:
    st.set_page_config(
        page_title="Bestands-Check: Hochgerechneter Bestand vs. erwartete Verkäufe",
        layout="wide",
    )

    st.title("Bestands-Check")
    st.caption("Hochgerechneter Bestand im Abgleich mit den erwarteten Verkäufen.")

    # === SIDEBAR: Upload, Kanal, kaskadierende Filter ===
    with st.sidebar:
        st.header("Einstellungen")

        uploaded_file = st.file_uploader(
            "CSV-Datei hochladen",
            type=["csv"],
            help="Datei aus dem BI-/Warenwirtschaftssystem exportieren.",
        )

        kanal_option = st.radio(
            "Kanal-Auswahl",
            options=["Nur Stationär", "Stationär + Online"],
            index=0,
            help="Steuert, ob nur stationäre Umsätze oder Stationär + Online betrachtet werden.",
        )

    if uploaded_file is None:
        st.info("Bitte eine CSV-Datei in der Sidebar hochladen.")
        return

    # Daten einlesen und vorbereiten
    try:
        raw_df = detect_and_read_csv(uploaded_file)
        df = prepare_data(raw_df)
    except Exception as exc:
        st.error(
            "Die Datei konnte nicht verarbeitet werden. "
            "Bitte Export und Spaltenstruktur prüfen."
        )
        st.exception(exc)
        return

    # --- kaskadierende Filter in der Sidebar ---
    with st.sidebar:
        st.subheader("Filter")

        filter_selections = {}
        temp_df = df.copy()

        for col in FILTER_COLS:
            if col in temp_df.columns:
                options = sorted(temp_df[col].dropna().unique().tolist())
                if not options:
                    continue
                selected = st.multiselect(
                    col,
                    options=options,
                    default=options,
                )
                filter_selections[col] = selected
                # Eingrenzung für nächste Filterdimension
                temp_df = temp_df[temp_df[col].isin(selected)]

    # Filter anwenden
    df_filtered = df.copy()
    for col, selected in filter_selections.items():
        df_filtered = df_filtered[df_filtered[col].isin(selected)]

    if df_filtered.empty:
        st.warning(
            "Es sind keine Daten für die aktuelle Filterkombination vorhanden. "
            "Bitte Filter in der Sidebar anpassen."
        )
        return

    # --- Datenvorschau ---
    with st.expander("Datenvorschau"):
        st.write(
            f"**Zeilen:** {len(df_filtered):,}".replace(",", "."),
            f"  |  **Spalten:** {df_filtered.shape[1]}",
        )
        st.dataframe(df_filtered.head(20), use_container_width=True)

    # === DIMENSIONEN für die Aggregation (über der Tabelle) ===
    st.subheader("Dimensionen für Gruppierung")

    available_dims = [
        "Produktkategorie",
        "Category",
        "Hauptwarengruppe",
        "Warengruppe",
        "Marke",
        "Filiale",
        "Kanal",
    ]

    group_cols = st.multiselect(
        "Dimensionen auswählen",
        options=[c for c in available_dims if c in df_filtered.columns],
        default=[c for c in DEFAULT_GROUP_COLS if c in df_filtered.columns],
        help="Steuert, über welche Dimensionen aggregiert wird. Weniger Dimensionen = weniger Detail / weniger Duplikate.",
    )

    # --- Aggregation nach gewählten Dimensionen ---
    agg_df = aggregate_data(df_filtered, kanal_option, group_cols)

    if agg_df.empty:
        st.warning(
            "Es konnten keine Aggregationen gebildet werden "
            "(kein Ist-/Vergleichszeitraum oder zu starke Filter)."
        )
        return

    # === Status-Filter (wirkt auf beide Tabellen & KPIs) ===
    st.subheader("Status-Filter")

    status_options = sorted(agg_df["status"].dropna().unique().tolist())
    status_selected = st.multiselect(
        "Status auswählen",
        options=status_options,
        default=status_options,
        help="z. B. nur Out-of-Stock und kritisch anzeigen.",
    )

    agg_df_view = agg_df[agg_df["status"].isin(status_selected)].copy()

    if agg_df_view.empty:
        st.warning(
            "Für die gewählte Status-Auswahl sind keine Daten vorhanden. "
            "Bitte Status-Filter anpassen."
        )
        return

    # --- KPIs auf Basis der gefilterten Aggregation ---
    kpis = compute_kpis(agg_df_view)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Summe hochgerechneter Bestand",
            format_int_de(kpis["sum_bestand"]),
        )
    with col2:
        st.metric(
            "Summe erwartete Verkäufe",
            format_int_de(kpis["sum_erwartet"]),
        )
    with col3:
        st.metric(
            "Ø Deckungsquote",
            format_percent_de(kpis["avg_deckungsquote"], decimals=1),
        )
    with col4:
        krit_text = (
            f"{kpis['count_oos']} Out-of-Stock / {kpis['count_kritisch']} kritisch"
        )
        st.metric("Risiko-Positionen", krit_text)

    # === Tabs: alle Positionen vs. Risikopositionen ===
    tab_all, tab_risk = st.tabs(["Alle Positionen", "Risikopositionen"])

    # --- Tab 1: alle Positionen ---
    with tab_all:
        st.subheader("Detailübersicht – alle Positionen")

        display_df_all = make_display_df(agg_df_view, group_cols)
        st.dataframe(display_df_all, use_container_width=True)

        csv_buffer_all = io.StringIO()
        export_cols_all = (group_cols or []) + [
            "hochgerechneter_bestand",
            "erwartete_verkaeufe",
            "deckungsquote_pct",
            "status",
        ]
        export_cols_all = [c for c in export_cols_all if c in agg_df_view.columns]
        export_df_all = agg_df_view[export_cols_all].copy()
        export_df_all.to_csv(csv_buffer_all, index=False)

        st.download_button(
            label="Alle Positionen als CSV herunterladen",
            data=csv_buffer_all.getvalue(),
            file_name="bestandsanalyse_alle.csv",
            mime="text/csv",
        )

    # --- Tab 2: nur Risikopositionen ---
    with tab_risk:
        st.subheader("Risikopositionen (Out-of-Stock & kritisch)")

        risk_df = agg_df_view[agg_df_view["status"].isin(["Out-of-Stock", "kritisch"])]

        if risk_df.empty:
            st.info("Aktuell gibt es keine Positionen mit Status Out-of-Stock oder kritisch.")
        else:
            display_df_risk = make_display_df(risk_df, group_cols)
            st.dataframe(display_df_risk, use_container_width=True)

            csv_buffer_risk = io.StringIO()
            export_cols_risk = (group_cols or []) + [
                "hochgerechneter_bestand",
                "erwartete_verkaeufe",
                "deckungsquote_pct",
                "status",
            ]
            export_cols_risk = [c for c in export_cols_risk if c in risk_df.columns]
            export_df_risk = risk_df[export_cols_risk].copy()
            export_df_risk.to_csv(csv_buffer_risk, index=False)

            st.download_button(
                label="Risikopositionen als CSV herunterladen",
                data=csv_buffer_risk.getvalue(),
                file_name="bestandsanalyse_risiko.csv",
                mime="text/csv",
            )

    # Hilfetext zur Logik
    with st.expander("Logik Hochrechnung & Vergleichszeitraum"):
        st.markdown(
            """
- Hochgerechneter Bestand = Bestand Menge (Vortag) + Offene Bestell. Menge – Umsatz Menge (Ist-Zeitraum).
- Vergleichszeitraum liefert die erwarteten Verkäufe über `Umsatz Menge`.
- Deckungsquote = Hochgerechneter Bestand / erwartete Verkäufe (in %).
- Ampel:
    - Rot = Out-of-Stock (Bestand ≤ 0)
    - Orange = kritisch (Bestand kleiner als erwartete Verkäufe)
    - Grün = ausreichend / Überdeckung.
            """
        )


if __name__ == "__main__":
    main()
