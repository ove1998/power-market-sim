"""
SMARD Data Importer

Importiert historische Strommarkt-Daten von SMARD.de (Bundesnetzagentur).
Verarbeitet CSV-Dateien und erstellt PyPSA-kompatible Zeitreihen.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import warnings

from ..utils.logging_config import get_logger, LogContext


logger = get_logger("data.smard_importer")


class SMARDImporter:
    """
    Importiert und verarbeitet SMARD CSV-Dateien.

    SMARD-Daten haben spezielle Formate:
    - Semikolon als Trennzeichen
    - Komma als Dezimaltrennzeichen
    - Deutsche Datumsformate
    """

    def __init__(self, raw_data_dir: Optional[Path] = None):
        """
        Initialisiert den Importer.

        Parameters:
        -----------
        raw_data_dir : Path, optional
            Verzeichnis mit SMARD CSV-Dateien
        """
        self.logger = get_logger("data.smard_importer")

        if raw_data_dir is None:
            # Standard: data/raw/smard/
            current_file = Path(__file__)
            project_root = current_file.parents[2]
            raw_data_dir = project_root / "data" / "raw" / "smard"

        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Output: data/processed/
        self.processed_dir = self.raw_data_dir.parents[1] / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"SMARDImporter initialized:")
        self.logger.info(f"  Raw data: {self.raw_data_dir}")
        self.logger.info(f"  Processed: {self.processed_dir}")

        # Installierte Kapazitäten (werden beim Import geladen)
        self.installed_capacities = {}

    def import_all(self, year: int = 2025) -> Dict[str, pd.DataFrame]:
        """
        Importiert alle verfügbaren SMARD-Daten für ein Jahr.

        Parameters:
        -----------
        year : int
            Jahr (z.B. 2025)

        Returns:
        --------
        dict
            Dictionary mit allen importierten Zeitreihen
        """
        with LogContext(self.logger, f"Importing SMARD data for {year}", log_memory=True):
            results = {}

            # 1. Installierte Kapazitäten laden (benötigt für CF-Berechnung)
            self.logger.info("Loading installed capacities...")
            self._load_installed_capacities(year)

            # 2. Demand (Nachfrage)
            self.logger.info("Importing demand...")
            demand = self.import_demand(year)
            if demand is not None:
                results['demand'] = demand

            # 3. Alle Erzeugungstechnologien
            technologies = [
                'wind_onshore',
                'wind_offshore',
                'solar',
                'nuclear',
                'lignite',
                'hard_coal',
                'gas',
                'hydro',
                'biomass',
                'pumped_hydro'
            ]

            for tech in technologies:
                self.logger.info(f"Importing {tech}...")
                data = self.import_generation(tech, year)
                if data is not None:
                    results[tech] = data

            # 4. Grenzüberschreitender Handel
            self.logger.info("Importing cross-border flows...")
            cross_border = self.import_cross_border(year)
            if cross_border is not None:
                results['cross_border'] = cross_border

            # 5. Day-Ahead Preise
            self.logger.info("Importing day-ahead prices...")
            prices = self.import_prices(year)
            if prices is not None:
                results['prices'] = prices

            self.logger.info(f"Successfully imported {len(results)} datasets")

            return results

    def _load_installed_capacities(self, year: int):
        """Lädt installierte Kapazitäten aus SMARD-Datei."""
        capacity_file = self.raw_data_dir / f"installed_capacity_{year}.csv"

        if not capacity_file.exists():
            self.logger.warning(f"Installed capacity file not found: {capacity_file}")
            self.logger.warning("Using default capacities from constants.py")
            return

        try:
            # SMARD Format: Semikolon-getrennt, Komma als Dezimaltrennzeichen
            df = pd.read_csv(
                capacity_file,
                sep=';',
                decimal=',',
                encoding='utf-8'
            )

            # Extrahiere Kapazitäten (Format variiert je nach SMARD-Datei)
            # Typische Spalten: Technologie, Installierte Leistung [MW]
            # TODO: Anpassen an tatsächliches SMARD-Format

            self.logger.info(f"Loaded installed capacities from {capacity_file}")

        except Exception as e:
            self.logger.error(f"Failed to load installed capacities: {e}")

    def import_demand(self, year: int) -> Optional[pd.DataFrame]:
        """
        Importiert Stromnachfrage (Realisierter Stromverbrauch).

        Parameters:
        -----------
        year : int
            Jahr

        Returns:
        --------
        pd.DataFrame
            Nachfrage mit Zeitstempel-Index und 'demand_mw' Spalte
        """
        demand_file = self.raw_data_dir / f"demand_{year}.csv"

        if not demand_file.exists():
            self.logger.warning(f"Demand file not found: {demand_file}")
            return None

        try:
            # SMARD Format
            df = self._read_smard_csv(demand_file)

            # Extrahiere Nachfrage-Spalte
            # SMARD-Format: "Gesamt (Netzlast) [MWh] Originalauflösungen"
            value_col = self._find_value_column(df, ['Gesamt', 'Netzlast', 'MWh'])

            if value_col is None:
                self.logger.error(f"Could not find demand column in {demand_file}")
                return None

            # Erstelle Clean DataFrame
            result = pd.DataFrame({
                'timestamp': df['timestamp'],
                'demand_mw': df[value_col]
            })

            result.set_index('timestamp', inplace=True)

            # Validierung
            self._validate_timeseries(result, 'demand')

            # Speichere als Parquet
            output_file = self.processed_dir / f"demand_germany_hourly.parquet"
            result.to_parquet(output_file)

            self.logger.info(
                f"Imported demand: {len(result)} hours, "
                f"avg={result['demand_mw'].mean():.0f} MW, "
                f"peak={result['demand_mw'].max():.0f} MW"
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to import demand: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def import_generation(self, technology: str, year: int) -> Optional[pd.DataFrame]:
        """
        Importiert Erzeugungsdaten und berechnet Kapazitätsfaktoren.

        Parameters:
        -----------
        technology : str
            'wind_onshore', 'wind_offshore', 'solar'
        year : int
            Jahr

        Returns:
        --------
        pd.DataFrame
            Erzeugung und CF mit Zeitstempel-Index
        """
        gen_file = self.raw_data_dir / f"{technology}_{year}.csv"

        if not gen_file.exists():
            self.logger.warning(f"Generation file not found: {gen_file}")
            return None

        try:
            # SMARD Format
            df = self._read_smard_csv(gen_file)

            # Extrahiere Erzeugungs-Spalte
            value_col = self._find_value_column(df, ['MW', 'MWh'])

            if value_col is None:
                self.logger.error(f"Could not find generation column in {gen_file}")
                return None

            # Erstelle Clean DataFrame
            result = pd.DataFrame({
                'timestamp': df['timestamp'],
                'generation_mw': df[value_col]
            })

            result.set_index('timestamp', inplace=True)

            # Berechne Kapazitätsfaktor
            capacity_mw = self._get_installed_capacity(technology, year)

            if capacity_mw > 0:
                result['capacity_factor'] = result['generation_mw'] / capacity_mw
                result['capacity_factor'] = result['capacity_factor'].clip(0, 1.2)  # Max 120% (Überbuchung)
            else:
                self.logger.warning(f"No installed capacity for {technology}, cannot calculate CF")
                result['capacity_factor'] = np.nan

            # Validierung
            self._validate_timeseries(result, technology)

            # Speichere als Parquet
            output_file = self.processed_dir / f"{technology}_germany_hourly.parquet"
            result.to_parquet(output_file)

            avg_cf = result['capacity_factor'].mean() if not result['capacity_factor'].isna().all() else 0
            self.logger.info(
                f"Imported {technology}: {len(result)} hours, "
                f"avg generation={result['generation_mw'].mean():.0f} MW, "
                f"avg CF={avg_cf:.2%}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to import {technology}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def import_prices(self, year: int) -> Optional[pd.DataFrame]:
        """
        Importiert Day-Ahead Strompreise.

        Parameters:
        -----------
        year : int
            Jahr

        Returns:
        --------
        pd.DataFrame
            Preise mit Zeitstempel-Index und 'price_eur_per_mwh' Spalte
        """
        price_file = self.raw_data_dir / f"day_ahead_prices_{year}.csv"

        if not price_file.exists():
            self.logger.warning(f"Price file not found: {price_file}")
            return None

        try:
            # SMARD Format
            df = self._read_smard_csv(price_file)

            # Extrahiere Preis-Spalte
            value_col = self._find_value_column(df, ['EUR', 'MWh', 'Preis'])

            if value_col is None:
                self.logger.error(f"Could not find price column in {price_file}")
                return None

            # Erstelle Clean DataFrame
            result = pd.DataFrame({
                'timestamp': df['timestamp'],
                'price_eur_per_mwh': df[value_col]
            })

            result.set_index('timestamp', inplace=True)

            # Validierung
            self._validate_timeseries(result, 'prices')

            # Speichere als Parquet
            output_file = self.processed_dir / f"day_ahead_prices_germany_hourly.parquet"
            result.to_parquet(output_file)

            self.logger.info(
                f"Imported prices: {len(result)} hours, "
                f"avg={result['price_eur_per_mwh'].mean():.2f} EUR/MWh, "
                f"min={result['price_eur_per_mwh'].min():.2f}, "
                f"max={result['price_eur_per_mwh'].max():.2f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to import prices: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def import_cross_border(self, year: int) -> Optional[pd.DataFrame]:
        """
        Importiert grenzüberschreitende Stromflüsse (Import/Export).

        Parameters:
        -----------
        year : int
            Jahr

        Returns:
        --------
        pd.DataFrame
            Stromflüsse mit Zeitstempel-Index
            Positiv = Import, Negativ = Export
        """
        cross_border_file = self.raw_data_dir / f"cross_border_{year}.csv"

        if not cross_border_file.exists():
            self.logger.warning(f"Cross-border file not found: {cross_border_file}")
            return None

        try:
            # SMARD Format
            df = self._read_smard_csv(cross_border_file)

            # Extrahiere Stromfluss-Spalte
            value_col = self._find_value_column(df, ['MW', 'MWh', 'Saldo'])

            if value_col is None:
                self.logger.error(f"Could not find cross-border column in {cross_border_file}")
                return None

            # Erstelle Clean DataFrame
            result = pd.DataFrame({
                'timestamp': df['timestamp'],
                'net_import_mw': df[value_col]  # Positiv = Import, Negativ = Export
            })

            result.set_index('timestamp', inplace=True)

            # Berechne separate Import/Export
            result['import_mw'] = result['net_import_mw'].clip(lower=0)
            result['export_mw'] = result['net_import_mw'].clip(upper=0).abs()

            # Validierung
            self._validate_timeseries(result, 'cross_border')

            # Speichere als Parquet
            output_file = self.processed_dir / f"cross_border_germany_hourly.parquet"
            result.to_parquet(output_file)

            avg_import = result['import_mw'].mean()
            avg_export = result['export_mw'].mean()
            net_import = result['net_import_mw'].mean()

            self.logger.info(
                f"Imported cross-border: {len(result)} hours, "
                f"avg import={avg_import:.0f} MW, "
                f"avg export={avg_export:.0f} MW, "
                f"net={net_import:.0f} MW"
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to import cross-border: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _read_smard_csv(self, filepath: Path) -> pd.DataFrame:
        """
        Liest SMARD CSV mit speziellem Format.

        SMARD Format:
        - Semikolon-getrennt
        - Komma als Dezimaltrennzeichen
        - Datum;Uhrzeit;Wert Format
        """
        # Lese mit Semikolon und Komma als Dezimal
        df = pd.read_csv(
            filepath,
            sep=';',
            decimal=',',
            encoding='utf-8',
            na_values=['-', '']
        )

        # Parse Zeitstempel
        # SMARD Format: Datum + Uhrzeit Spalten oder kombiniert
        if 'Datum' in df.columns and 'Uhrzeit' in df.columns:
            # Kombiniere Datum und Uhrzeit
            df['timestamp'] = pd.to_datetime(
                df['Datum'] + ' ' + df['Uhrzeit'],
                format='%d.%m.%Y %H:%M',
                errors='coerce'
            )
        elif 'Datum' in df.columns:
            # Nur Datum (versuche zu parsen)
            df['timestamp'] = pd.to_datetime(
                df['Datum'],
                format='%d.%m.%Y %H:%M',
                errors='coerce'
            )
        else:
            # Versuche erste Spalte als Zeitstempel
            df['timestamp'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')

        # Entferne Zeilen ohne gültigen Zeitstempel
        df = df.dropna(subset=['timestamp'])

        # Konvertiere zu UTC (SMARD ist MEZ/MESZ)
        # Deutschland = Europe/Berlin
        df['timestamp'] = df['timestamp'].dt.tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='shift_forward')
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

        return df

    def _find_value_column(self, df: pd.DataFrame, keywords: list) -> Optional[str]:
        """Findet die Wert-Spalte basierend auf Keywords."""
        for col in df.columns:
            if col == 'timestamp':
                continue
            # Prüfe ob einer der Keywords im Spaltennamen vorkommt
            if any(kw.lower() in col.lower() for kw in keywords):
                return col
        return None

    def _get_installed_capacity(self, technology: str, year: int) -> float:
        """
        Gibt installierte Kapazität für eine Technologie zurück.

        Falls nicht aus Datei geladen, verwende Defaults.
        """
        # Mapping von unseren Namen zu SMARD-Namen
        capacity_mapping = {
            'wind_onshore': 'Wind Onshore',
            'wind_offshore': 'Wind Offshore',
            'solar': 'Photovoltaik',
            'nuclear': 'Kernenergie',
            'lignite': 'Braunkohle',
            'hard_coal': 'Steinkohle',
            'gas': 'Erdgas',
            'hydro': 'Wasserkraft',
            'biomass': 'Biomasse',
            'pumped_hydro': 'Pumpspeicher'
        }

        smard_name = capacity_mapping.get(technology)

        if smard_name and smard_name in self.installed_capacities:
            return self.installed_capacities[smard_name]

        # Fallback: Defaults aus 2024/2025 Deutschland (ungefähre Werte)
        # Quelle: Bundesnetzagentur, Energy Charts
        default_capacities = {
            # Erneuerbare
            'wind_onshore': 60000,   # 60 GW
            'wind_offshore': 10000,  # 10 GW
            'solar': 85000,          # 85 GW (stark gewachsen!)
            'hydro': 5500,           # 5.5 GW (Run-of-River + Reservoir)
            'biomass': 9000,         # 9 GW
            'pumped_hydro': 9500,    # 9.5 GW

            # Konventionell
            'nuclear': 0,            # 0 GW (seit 2023 abgeschaltet!)
            'lignite': 18000,        # 18 GW (wird reduziert)
            'hard_coal': 22000,      # 22 GW (wird reduziert)
            'gas': 32000,            # 32 GW (CCGT + OCGT)
        }

        capacity = default_capacities.get(technology, 0)

        if capacity > 0:
            self.logger.warning(
                f"Using default capacity for {technology}: {capacity/1000:.1f} GW"
            )

        return capacity

    def _validate_timeseries(self, df: pd.DataFrame, name: str):
        """Validiert Zeitreihe auf Vollständigkeit und Qualität."""
        # Prüfe auf Lücken
        if len(df) == 0:
            self.logger.error(f"{name}: Empty timeseries!")
            return

        # Erwartete Zeitschritte (stündlich)
        expected_freq = pd.Timedelta(hours=1)
        time_diffs = df.index.to_series().diff()

        # Prüfe auf große Lücken (> 2 Stunden)
        large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]

        if len(large_gaps) > 0:
            self.logger.warning(
                f"{name}: Found {len(large_gaps)} gaps > 2 hours. "
                f"Largest gap: {large_gaps.max()}"
            )

        # Prüfe auf fehlende Werte
        for col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                pct_missing = 100 * n_missing / len(df)
                self.logger.warning(
                    f"{name}.{col}: {n_missing} missing values ({pct_missing:.1f}%)"
                )

        # Prüfe auf negative Werte (sollte nicht vorkommen)
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                n_negative = (df[col] < 0).sum()
                if n_negative > 0:
                    self.logger.warning(
                        f"{name}.{col}: {n_negative} negative values"
                    )

    def get_summary_statistics(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Erstellt Zusammenfassung der importierten Daten.

        Parameters:
        -----------
        data : dict
            Dictionary mit DataFrames

        Returns:
        --------
        pd.DataFrame
            Statistik-Tabelle
        """
        stats = []

        for name, df in data.items():
            for col in df.columns:
                stat = {
                    'dataset': name,
                    'variable': col,
                    'count': len(df),
                    'missing': df[col].isna().sum(),
                    'mean': df[col].mean() if df[col].dtype in [np.float64, np.int64] else np.nan,
                    'std': df[col].std() if df[col].dtype in [np.float64, np.int64] else np.nan,
                    'min': df[col].min() if df[col].dtype in [np.float64, np.int64] else np.nan,
                    'max': df[col].max() if df[col].dtype in [np.float64, np.int64] else np.nan,
                }
                stats.append(stat)

        return pd.DataFrame(stats)


# CLI
if __name__ == "__main__":
    from ..utils.logging_config import setup_logging
    import sys

    setup_logging(level="INFO", console=True)

    # Jahr aus Command-Line
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025

    logger.info(f"Starting SMARD import for {year}")

    # Importiere Daten
    importer = SMARDImporter()
    data = importer.import_all(year=year)

    # Statistiken
    if len(data) > 0:
        stats = importer.get_summary_statistics(data)

        print("\n" + "=" * 80)
        print("IMPORT SUMMARY")
        print("=" * 80)
        print(stats.to_string(index=False))

        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"Imported {len(data)} datasets to data/processed/")
    else:
        print("\n" + "=" * 80)
        print("ERROR: No data imported!")
        print("=" * 80)
        print("Please check:")
        print("1. CSV files in data/raw/smard/ exist")
        print("2. File names match pattern: <type>_2025.csv")
        print("3. Files have correct SMARD CSV format")
