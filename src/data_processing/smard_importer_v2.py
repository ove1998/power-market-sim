"""
SMARD Data Importer V2

Verbesserte Version, die echte SMARD-Dateinamen erkennt und
aggregierte Erzeugungsdateien verarbeitet.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import re

from ..utils.logging_config import get_logger, LogContext


logger = get_logger("data.smard_importer_v2")


class SMARDImporterV2:
    """
    Verbesserter SMARD Importer für echte SMARD-Dateinamen.

    Erkennt automatisch:
    - Realisierte_Erzeugung (enthält alle Technologien in Spalten)
    - Realisierter_Stromverbrauch
    - Gro_handelspreise
    - Installierte_Erzeugungsleistung
    """

    def __init__(self, raw_data_dir: Optional[Path] = None):
        """Initialisiert den Importer."""
        self.logger = get_logger("data.smard_importer_v2")

        if raw_data_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parents[2]
            raw_data_dir = project_root / "data" / "raw" / "smard"

        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        self.processed_dir = self.raw_data_dir.parents[1] / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"SMARDImporterV2 initialized:")
        self.logger.info(f"  Raw data: {self.raw_data_dir}")
        self.logger.info(f"  Processed: {self.processed_dir}")

        # Mapping SMARD-Spaltennamen → unsere Namen
        self.technology_mapping = {
            # Erneuerbare
            'Wind Onshore': 'wind_onshore',
            'Wind Offshore': 'wind_offshore',
            'Photovoltaik': 'solar',
            'Sonstige Erneuerbare': 'other_renewables',
            'Wasserkraft': 'hydro',
            'Biomasse': 'biomass',

            # Konventionelle
            'Kernenergie': 'nuclear',
            'Braunkohle': 'lignite',
            'Steinkohle': 'hard_coal',
            'Erdgas': 'gas',
            'Pumpspeicher': 'pumped_hydro',
            'Sonstige Konventionelle': 'other_conventional',
        }

        # Installierte Kapazitäten (werden beim Import geladen)
        self.installed_capacities = {}

    def find_files(self) -> Dict[str, Path]:
        """
        Findet alle SMARD-Dateien im raw_data_dir.

        Returns:
        --------
        dict
            Mapping von Dateityp → Pfad
        """
        files = {}

        for file in self.raw_data_dir.glob("*.csv"):
            filename = file.stem.lower()

            if 'stromverbrauch' in filename or 'verbrauch' in filename:
                files['demand'] = file
            elif 'erzeugung' in filename and 'installiert' not in filename:
                files['generation'] = file
            elif 'handelspreise' in filename or 'preise' in filename or 'auktion' in filename:
                files['prices'] = file
            elif 'installiert' in filename or 'leistung' in filename:
                files['capacity'] = file
            elif 'grenz' in filename or 'handel' in filename or 'saldo' in filename:
                files['cross_border'] = file

        self.logger.info(f"Found {len(files)} SMARD files:")
        for ftype, fpath in files.items():
            self.logger.info(f"  - {ftype}: {fpath.name}")

        return files

    def import_all(self, year: int = 2025) -> Dict[str, pd.DataFrame]:
        """
        Importiert alle verfügbaren SMARD-Daten.

        Returns:
        --------
        dict
            Dictionary mit allen importierten Zeitreihen
        """
        with LogContext(self.logger, f"Importing SMARD data", log_memory=True):
            results = {}

            # Finde alle Dateien
            files = self.find_files()

            if len(files) == 0:
                self.logger.error("No SMARD files found!")
                self.logger.error(f"Please place CSV files in: {self.raw_data_dir}")
                return results

            # 1. Installierte Kapazitäten laden
            if 'capacity' in files:
                self.logger.info("Loading installed capacities...")
                self._load_installed_capacities(files['capacity'])

            # 2. Demand
            if 'demand' in files:
                self.logger.info("Importing demand...")
                demand = self.import_demand(files['demand'])
                if demand is not None:
                    results['demand'] = demand

            # 3. Erzeugung (enthält alle Technologien!)
            if 'generation' in files:
                self.logger.info("Importing generation (all technologies)...")
                generation = self.import_generation_aggregated(files['generation'])
                if generation is not None:
                    results.update(generation)  # Füge alle Technologien hinzu

            # 4. Preise
            if 'prices' in files:
                self.logger.info("Importing prices...")
                prices = self.import_prices(files['prices'])
                if prices is not None:
                    results['prices'] = prices

            # 5. Cross-Border (falls vorhanden)
            if 'cross_border' in files:
                self.logger.info("Importing cross-border...")
                cross_border = self.import_cross_border(files['cross_border'])
                if cross_border is not None:
                    results['cross_border'] = cross_border

            self.logger.info(f"Successfully imported {len(results)} datasets")

            return results

    def import_demand(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Importiert Demand aus SMARD-Datei."""
        try:
            df = self._read_smard_csv(filepath)

            # Finde Wert-Spalte
            value_col = self._find_value_column(df, ['Gesamt', 'Netzlast', 'Verbrauch'])

            if value_col is None:
                self.logger.error(f"Could not find demand column in {filepath.name}")
                return None

            result = pd.DataFrame({
                'timestamp': df['timestamp'],
                'demand_mw': df[value_col]
            })

            result.set_index('timestamp', inplace=True)

            # Speichere
            output_file = self.processed_dir / "demand_germany_hourly.parquet"
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

    def import_generation_aggregated(self, filepath: Path) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Importiert aggregierte Erzeugungsdatei mit allen Technologien.

        Die SMARD-Datei enthält alle Technologien als Spalten:
        - Datum | Uhrzeit | Biomasse | Wasserkraft | Wind Offshore | ...

        Returns:
        --------
        dict
            Dictionary mit technology → DataFrame
        """
        try:
            df = self._read_smard_csv(filepath)

            self.logger.info(f"Found {len(df.columns)} columns in generation file")

            # Zeige alle Spalten (für Debugging)
            self.logger.info("Columns: " + ", ".join([c for c in df.columns if c != 'timestamp']))

            results = {}

            # Durchsuche alle Spalten nach Technologien
            for col in df.columns:
                if col == 'timestamp':
                    continue

                # Finde passende Technologie
                tech_name = None
                for smard_name, our_name in self.technology_mapping.items():
                    if smard_name.lower() in col.lower():
                        tech_name = our_name
                        break

                if tech_name is None:
                    self.logger.warning(f"Unknown technology column: {col}")
                    continue

                # Extrahiere Daten (MWh für die Stunde = MW Durchschnittsleistung)
                generation_mw = df[col].values  # Use .values to avoid index alignment issues

                # Berechne Kapazitätsfaktor
                capacity_mw = self._get_installed_capacity(tech_name)

                if capacity_mw > 0:
                    capacity_factor = generation_mw / capacity_mw
                    capacity_factor = np.clip(capacity_factor, 0, 1.5)  # Max 150%
                else:
                    capacity_factor = np.full_like(generation_mw, np.nan, dtype=float)

                # Erstelle DataFrame mit UTC-Timestamp-Index
                result = pd.DataFrame({
                    'generation_mw': generation_mw,
                    'capacity_factor': capacity_factor
                }, index=df['timestamp'])  # Keep timezone-aware index

                # Speichere
                output_file = self.processed_dir / f"{tech_name}_germany_hourly.parquet"
                result.to_parquet(output_file)

                avg_gen = result['generation_mw'].mean()
                avg_cf = result['capacity_factor'].mean() if not result['capacity_factor'].isna().all() else 0

                self.logger.info(
                    f"Imported {tech_name}: avg={avg_gen:.0f} MW, CF={avg_cf:.2%}"
                )

                results[tech_name] = result

            return results

        except Exception as e:
            self.logger.error(f"Failed to import generation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def import_prices(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Importiert Day-Ahead Preise."""
        try:
            df = self._read_smard_csv(filepath)

            # Finde Preis-Spalte
            value_col = self._find_value_column(df, ['EUR', 'MWh', 'Preis', 'Deutschland'])

            if value_col is None:
                self.logger.error(f"Could not find price column in {filepath.name}")
                return None

            result = pd.DataFrame({
                'timestamp': df['timestamp'],
                'price_eur_per_mwh': df[value_col]
            })

            result.set_index('timestamp', inplace=True)

            # Speichere
            output_file = self.processed_dir / "day_ahead_prices_germany_hourly.parquet"
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

    def import_cross_border(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Importiert grenzüberschreitende Flüsse."""
        try:
            df = self._read_smard_csv(filepath)

            # Finde Saldo-Spalte
            value_col = self._find_value_column(df, ['Saldo', 'Gesamt', 'MW', 'MWh'])

            if value_col is None:
                self.logger.error(f"Could not find cross-border column in {filepath.name}")
                return None

            result = pd.DataFrame({
                'net_import_mw': df[value_col]
            }, index=df['timestamp'])

            # Berechne Import/Export
            result['import_mw'] = result['net_import_mw'].clip(lower=0)
            result['export_mw'] = result['net_import_mw'].clip(upper=0).abs()

            # Speichere
            output_file = self.processed_dir / "cross_border_germany_hourly.parquet"
            result.to_parquet(output_file)

            self.logger.info(
                f"Imported cross-border: {len(result)} hours, "
                f"net={result['net_import_mw'].mean():.0f} MW"
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to import cross-border: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _load_installed_capacities(self, filepath: Path):
        """Lädt installierte Kapazitäten."""
        try:
            df = self._read_smard_csv(filepath)

            # Versuche Kapazitäten zu extrahieren
            # SMARD-Format variiert, daher flexibel
            for col in df.columns:
                if col == 'timestamp':
                    continue

                for smard_name, our_name in self.technology_mapping.items():
                    if smard_name.lower() in col.lower():
                        # Nimm letzten nicht-NaN Wert
                        capacity = df[col].dropna().iloc[-1] if len(df[col].dropna()) > 0 else 0
                        self.installed_capacities[our_name] = capacity
                        self.logger.info(f"Loaded capacity {our_name}: {capacity:.0f} MW")
                        break

        except Exception as e:
            self.logger.warning(f"Failed to load capacities: {e}")

    def _read_smard_csv(self, filepath: Path) -> pd.DataFrame:
        """Liest SMARD CSV mit flexiblem Format."""
        # Versuche verschiedene Encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(
                    filepath,
                    sep=';',
                    decimal=',',
                    encoding=encoding,
                    na_values=['-', ''],
                    thousands='.'
                )
                break
            except:
                continue
        else:
            raise ValueError(f"Could not read {filepath} with any encoding")

        # Parse Zeitstempel
        if 'Datum von' in df.columns:
            # Neueres SMARD-Format mit "Datum von" und "Datum bis"
            df['timestamp'] = pd.to_datetime(
                df['Datum von'],
                format='%d.%m.%Y %H:%M',
                errors='coerce'
            )
        elif 'Datum' in df.columns:
            if 'Uhrzeit' in df.columns:
                # Kombiniere Datum + Uhrzeit
                df['timestamp'] = pd.to_datetime(
                    df['Datum'] + ' ' + df['Uhrzeit'],
                    format='%d.%m.%Y %H:%M',
                    errors='coerce'
                )
            else:
                df['timestamp'] = pd.to_datetime(df['Datum'], errors='coerce')
        else:
            # Erste Spalte als Zeitstempel
            df['timestamp'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')

        # Entferne NaN Timestamps
        df = df.dropna(subset=['timestamp'])

        # Konvertiere zu UTC
        try:
            df['timestamp'] = df['timestamp'].dt.tz_localize(
                'Europe/Berlin',
                ambiguous='infer',
                nonexistent='shift_forward'
            )
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        except:
            # Falls schon UTC oder andere Probleme
            pass

        return df

    def _find_value_column(self, df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        """Findet Wert-Spalte basierend auf Keywords."""
        for col in df.columns:
            if col == 'timestamp':
                continue
            col_lower = col.lower()
            if any(kw.lower() in col_lower for kw in keywords):
                return col
        return None

    def _get_installed_capacity(self, technology: str) -> float:
        """Gibt installierte Kapazität zurück."""
        if technology in self.installed_capacities:
            return self.installed_capacities[technology]

        # Fallback: Defaults
        default_capacities = {
            'wind_onshore': 60000,
            'wind_offshore': 10000,
            'solar': 85000,
            'hydro': 5500,
            'biomass': 9000,
            'pumped_hydro': 9500,
            'nuclear': 0,
            'lignite': 18000,
            'hard_coal': 22000,
            'gas': 32000,
        }

        capacity = default_capacities.get(technology, 0)

        if capacity > 0:
            self.logger.warning(
                f"Using default capacity for {technology}: {capacity/1000:.1f} GW"
            )

        return capacity


# CLI
if __name__ == "__main__":
    from ..utils.logging_config import setup_logging
    import sys

    setup_logging(level="INFO", console=True)

    # Importiere Daten
    importer = SMARDImporterV2()
    data = importer.import_all()

    if len(data) > 0:
        print("\n" + "=" * 80)
        print("IMPORT SUCCESS!")
        print("=" * 80)
        print(f"\nImported {len(data)} datasets:")
        for name in sorted(data.keys()):
            print(f"  ✓ {name}")
        print(f"\nProcessed files saved to: {importer.processed_dir}")
    else:
        print("\n" + "=" * 80)
        print("ERROR: No data imported!")
        print("=" * 80)
