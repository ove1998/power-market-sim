"""
Cross-Border Flow and Price Processor

Verarbeitet physikalische Stromflüsse und Nachbarland-Preise aus SMARD-Daten
um realistische Import/Export-Preise zu berechnen.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import re

from ..utils.logging_config import get_logger

logger = get_logger("data.cross_border")


class CrossBorderProcessor:
    """
    Verarbeitet grenzüberschreitende Stromflüsse und Preise.

    Berechnet gewichtete Import/Export-Preise basierend auf:
    - Tatsächlichen physikalischen Flüssen pro Land
    - Day-Ahead Preisen der Nachbarländer
    """

    def __init__(self, raw_data_dir: Optional[Path] = None):
        """Initialisiert den Processor."""
        self.logger = get_logger("data.cross_border")

        if raw_data_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parents[2]
            raw_data_dir = project_root / "data" / "raw" / "smard"

        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = self.raw_data_dir.parents[1] / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Länder-Mapping (SMARD-Name → unsere Namen)
        self.countries = [
            'Belgien', 'Dänemark 1', 'Dänemark 2', 'Frankreich',
            'Niederlande', 'Norwegen 2', 'Österreich', 'Polen',
            'Schweden 4', 'Schweiz', 'Tschechien'
        ]

    def import_physical_flows(self) -> Optional[pd.DataFrame]:
        """
        Importiert physikalische Stromflüsse.

        Returns:
        --------
        pd.DataFrame
            Zeitreihe mit Export/Import pro Land
        """
        flow_file = self._find_file('physikalisch', 'stromfluss', 'fluss')

        if flow_file is None:
            self.logger.error("No physical flow file found!")
            return None

        try:
            self.logger.info(f"Importing physical flows from {flow_file.name}")

            # Lese CSV
            df = self._read_smard_csv(flow_file)

            # Extrahiere Flüsse pro Land
            flows = {'timestamp': df['timestamp']}

            # Nettoexport (Gesamt)
            net_export_col = self._find_column(df, ['Nettoexport'])
            if net_export_col:
                flows['net_export_mw'] = df[net_export_col].values

            # Export/Import pro Land
            for country in self.countries:
                export_col = self._find_column(df, [country, 'Export'])
                import_col = self._find_column(df, [country, 'Import'])

                if export_col and import_col:
                    # Netto-Fluss: Export - Import
                    # Positiv = Export, Negativ = Import
                    net_flow = df[export_col].fillna(0) - df[import_col].fillna(0).abs()
                    flows[f'{country}_net_mw'] = net_flow.values

            result = pd.DataFrame(flows)
            result.set_index('timestamp', inplace=True)

            # Speichere
            output_file = self.processed_dir / "cross_border_flows_hourly.parquet"
            result.to_parquet(output_file)

            self.logger.info(
                f"Imported physical flows: {len(result)} hours, "
                f"avg net export: {result['net_export_mw'].mean():.0f} MW"
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to import physical flows: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def import_neighbor_prices(self) -> Optional[pd.DataFrame]:
        """
        Importiert Day-Ahead Preise der Nachbarländer.

        Returns:
        --------
        pd.DataFrame
            Zeitreihe mit Preisen pro Land
        """
        price_file = self._find_file('handelspreise', 'preise')

        if price_file is None:
            self.logger.error("No price file found!")
            return None

        try:
            self.logger.info(f"Importing neighbor prices from {price_file.name}")

            # Lese CSV
            df = self._read_smard_csv(price_file)

            # Extrahiere Preise pro Land
            prices = {'timestamp': df['timestamp']}

            for country in self.countries:
                # Suche nach Spalte mit Format: "Belgien [€/MWh] ..."
                price_col = self._find_column(df, [country, '€/MWh'])

                if price_col:
                    prices[f'{country}_price'] = df[price_col].values

            result = pd.DataFrame(prices)
            result.set_index('timestamp', inplace=True)

            # Speichere
            output_file = self.processed_dir / "neighbor_prices_hourly.parquet"
            result.to_parquet(output_file)

            self.logger.info(
                f"Imported neighbor prices: {len(result)} hours, "
                f"{len([c for c in result.columns])} countries"
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to import neighbor prices: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def calculate_weighted_import_export_prices(
        self,
        flows: pd.DataFrame,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Berechnet gewichtete Import/Export-Preise.

        Parameters:
        -----------
        flows : pd.DataFrame
            Physikalische Flüsse pro Land (MW)
        prices : pd.DataFrame
            Preise pro Land (EUR/MWh)

        Returns:
        --------
        pd.DataFrame
            Zeitreihe mit gewichteten Import/Export-Preisen
        """
        self.logger.info("Calculating weighted import/export prices...")

        result = pd.DataFrame(index=flows.index)

        import_prices = []
        export_prices = []
        import_volumes = []
        export_volumes = []

        for ts in flows.index:
            # Sammle alle Flows und Preise für diesen Zeitpunkt
            imports = []  # (volume, price)
            exports = []  # (volume, price)

            for country in self.countries:
                flow_col = f'{country}_net_mw'
                price_col = f'{country}_price'

                if flow_col not in flows.columns or price_col not in prices.columns:
                    continue

                flow = flows.loc[ts, flow_col]
                price = prices.loc[ts, price_col]

                # Skip NaN
                if pd.isna(flow) or pd.isna(price):
                    continue

                if flow < -10:  # Import (negativ)
                    imports.append((abs(flow), price))
                elif flow > 10:  # Export (positiv)
                    exports.append((flow, price))

            # Berechne gewichtete Preise
            if imports:
                total_import_vol = sum(vol for vol, _ in imports)
                weighted_import_price = sum(vol * price for vol, price in imports) / total_import_vol
                import_prices.append(weighted_import_price)
                import_volumes.append(total_import_vol)
            else:
                import_prices.append(np.nan)
                import_volumes.append(0)

            if exports:
                total_export_vol = sum(vol for vol, _ in exports)
                weighted_export_price = sum(vol * price for vol, price in exports) / total_export_vol
                export_prices.append(weighted_export_price)
                export_volumes.append(total_export_vol)
            else:
                export_prices.append(np.nan)
                export_volumes.append(0)

        result['import_price_eur_per_mwh'] = import_prices
        result['export_price_eur_per_mwh'] = export_prices
        result['import_volume_mw'] = import_volumes
        result['export_volume_mw'] = export_volumes

        # Interpoliere fehlende Preise (falls nur wenig fehlt)
        result['import_price_eur_per_mwh'] = result['import_price_eur_per_mwh'].interpolate(
            method='linear', limit=3
        )
        result['export_price_eur_per_mwh'] = result['export_price_eur_per_mwh'].interpolate(
            method='linear', limit=3
        )

        # Speichere
        output_file = self.processed_dir / "weighted_import_export_prices_hourly.parquet"
        result.to_parquet(output_file)

        self.logger.info(
            f"Calculated weighted prices:\n"
            f"  Import price: {result['import_price_eur_per_mwh'].mean():.2f} ± "
            f"{result['import_price_eur_per_mwh'].std():.2f} EUR/MWh\n"
            f"  Export price: {result['export_price_eur_per_mwh'].mean():.2f} ± "
            f"{result['export_price_eur_per_mwh'].std():.2f} EUR/MWh\n"
            f"  Avg import volume: {result['import_volume_mw'].mean():.0f} MW\n"
            f"  Avg export volume: {result['export_volume_mw'].mean():.0f} MW"
        )

        return result

    def process_all(self) -> Dict[str, pd.DataFrame]:
        """
        Importiert und verarbeitet alle Cross-Border-Daten.

        Returns:
        --------
        dict
            Dictionary mit allen verarbeiteten DataFrames
        """
        results = {}

        # 1. Physikalische Flüsse
        flows = self.import_physical_flows()
        if flows is not None:
            results['flows'] = flows

        # 2. Nachbarland-Preise
        prices = self.import_neighbor_prices()
        if prices is not None:
            results['prices'] = prices

        # 3. Gewichtete Import/Export-Preise
        if flows is not None and prices is not None:
            weighted_prices = self.calculate_weighted_import_export_prices(flows, prices)
            results['weighted_prices'] = weighted_prices

        return results

    def _find_file(self, *keywords) -> Optional[Path]:
        """Findet Datei basierend auf Keywords."""
        for file in self.raw_data_dir.glob("*.csv"):
            filename_lower = file.stem.lower()
            if all(kw.lower() in filename_lower for kw in keywords):
                return file
        return None

    def _find_column(self, df: pd.DataFrame, keywords: list) -> Optional[str]:
        """Findet Spalte basierend auf Keywords."""
        for col in df.columns:
            if col == 'timestamp':
                continue
            col_lower = col.lower()
            if all(kw.lower() in col_lower for kw in keywords):
                return col
        return None

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
            df['timestamp'] = pd.to_datetime(
                df['Datum von'],
                format='%d.%m.%Y %H:%M',
                errors='coerce'
            )
        elif 'Datum' in df.columns:
            if 'Uhrzeit' in df.columns:
                df['timestamp'] = pd.to_datetime(
                    df['Datum'] + ' ' + df['Uhrzeit'],
                    format='%d.%m.%Y %H:%M',
                    errors='coerce'
                )
            else:
                df['timestamp'] = pd.to_datetime(df['Datum'], errors='coerce')
        else:
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
            pass

        return df


# CLI
if __name__ == "__main__":
    from ..utils.logging_config import setup_logging
    import sys

    setup_logging(level="INFO", console=True)

    # Verarbeite Cross-Border-Daten
    processor = CrossBorderProcessor()
    results = processor.process_all()

    if len(results) > 0:
        print("\n" + "=" * 80)
        print("CROSS-BORDER PROCESSING SUCCESS!")
        print("=" * 80)
        print(f"\nProcessed {len(results)} datasets:")
        for name in sorted(results.keys()):
            print(f"  ✓ {name}")
        print(f"\nProcessed files saved to: {processor.processed_dir}")
    else:
        print("\n" + "=" * 80)
        print("ERROR: No data processed!")
        print("=" * 80)
