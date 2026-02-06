"""
Network Builder for Power Market Simulation

Erstellt ein vereinfachtes Copper Plate Netzwerk für Deutschland.
Keine geografischen Constraints - nur Merit Order Dispatch basierend auf Kosten.
"""

import pypsa
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import yaml

from ..utils.logging_config import get_logger, LogContext
from ..utils.constants import (
    GENERATOR_TECHNOLOGIES,
    STORAGE_TECHNOLOGIES,
    DEFAULT_CAPACITIES_GW,
    DEFAULT_DEMAND,
    INTERCONNECTOR_PARAMS,
    UNITS,
    CO2_PRICE,
    calculate_marginal_cost_with_co2
)


class NetworkBuilder:
    """
    Baut ein Copper Plate Strommarkt-Netzwerk mit PyPSA.

    Copper Plate Modell:
    - Ein einziger Bus (Deutschland)
    - Keine Leitungsbeschränkungen
    - Merit Order Dispatch nur durch marginale Kosten
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        start_date: str = "2024-01-01",
        end_date: str = "2024-03-31",
        co2_price: Optional[float] = None
    ):
        """
        Initialisiert den Network Builder.

        Parameters:
        -----------
        config_path : str, optional
            Pfad zur Konfigurationsdatei
        start_date : str
            Start-Datum der Simulation (YYYY-MM-DD)
        end_date : str
            End-Datum der Simulation (YYYY-MM-DD)
        co2_price : float, optional
            CO2-Preis in EUR/t. Falls None, wird CO2_PRICE['default'] verwendet.
        """
        self.logger = get_logger("network.builder")

        # CO2-Preis
        self.co2_price = co2_price if co2_price is not None else CO2_PRICE['default']
        self.logger.info(f"Using CO2 price: {self.co2_price:.2f} EUR/t CO2")

        # Lade Konfiguration
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()

        # Zeitraum (mit UTC Timezone für Kompatibilität mit SMARD-Daten)
        self.start_date = pd.Timestamp(start_date, tz='UTC')
        self.end_date = pd.Timestamp(end_date, tz='UTC')
        self.snapshots = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='h',
            tz='UTC'
        )

        self.logger.info(
            f"NetworkBuilder initialized: {self.start_date} to {self.end_date} "
            f"({len(self.snapshots)} snapshots)"
        )

        # PyPSA Network
        self.network = None

    def _get_default_config(self) -> Dict:
        """Gibt Standard-Konfiguration zurück."""
        return {
            'network': {
                'name': 'Germany Copper Plate',
                'model_type': 'copper_plate'
            },
            'generators': DEFAULT_CAPACITIES_GW,
            'demand': DEFAULT_DEMAND,
            'interconnectors': INTERCONNECTOR_PARAMS
        }

    def build(self, scenario_config: Optional[Dict] = None) -> pypsa.Network:
        """
        Baut das komplette Netzwerk.

        Parameters:
        -----------
        scenario_config : dict, optional
            Szenario-spezifische Konfiguration (überschreibt Defaults)

        Returns:
        --------
        pypsa.Network
            Fertig konfiguriertes PyPSA-Netzwerk
        """
        with LogContext(self.logger, "Building Copper Plate Network", log_memory=True):
            # Neues leeres Netzwerk
            self.network = pypsa.Network()
            self.network.set_snapshots(self.snapshots)

            # Merge scenario config
            if scenario_config:
                self._merge_scenario_config(scenario_config)

            # Komponenten hinzufügen
            self._add_buses()
            self._add_demand()
            self._add_generators()
            self._add_interconnectors()

            # Optional: Speicher (falls im Szenario definiert)
            if scenario_config and 'storage' in scenario_config:
                self._add_storage(scenario_config['storage'])

            self.logger.info(f"Network built successfully: {self._get_network_summary()}")

            return self.network

    def _merge_scenario_config(self, scenario_config: Dict):
        """Merged Szenario-Konfiguration in die Basis-Konfiguration."""
        for key, value in scenario_config.items():
            if key in self.config and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value

    def _add_buses(self):
        """
        Fügt Busse hinzu.

        Copper Plate: Ein einziger Bus für ganz Deutschland.
        """
        self.network.add(
            "Bus",
            "DE",
            carrier="AC",
            x=10.0,  # Longitude (Deutschland Zentrum)
            y=51.0   # Latitude
        )

        self.logger.info("Added bus: DE (Copper Plate)")

    def _add_demand(self):
        """
        Fügt Nachfrage-Zeitreihe hinzu.

        Lädt oder generiert eine stündliche Nachfrage-Zeitreihe für Deutschland.
        """
        # Versuche, vorhandene Nachfrage-Zeitreihe zu laden
        demand_timeseries = self._load_demand_timeseries()

        if demand_timeseries is None:
            # Fallback: Generiere synthetische Nachfrage
            self.logger.warning("No demand timeseries found, generating synthetic demand")
            demand_timeseries = self._generate_synthetic_demand()

        # Skaliere Nachfrage, falls demand_scale im Config vorhanden
        demand_scale = self.config.get('demand_scale', 1.0)
        if demand_scale != 1.0:
            demand_timeseries = demand_timeseries * demand_scale
            self.logger.info(f"Demand scaled by factor {demand_scale:.2f}")

        # Füge Last-Komponente hinzu
        self.network.add(
            "Load",
            "demand_DE",
            bus="DE",
            p_set=demand_timeseries
        )

        avg_demand = demand_timeseries.mean()
        peak_demand = demand_timeseries.max()

        self.logger.info(
            f"Added demand: avg={avg_demand:.1f} MW, peak={peak_demand:.1f} MW"
        )

    def _load_demand_timeseries(self) -> Optional[pd.Series]:
        """
        Lädt historische Nachfrage-Zeitreihe.

        Returns:
        --------
        pd.Series or None
            Nachfrage in MW für jeden Snapshot
        """
        # Suche nach Demand-Daten im data/processed Verzeichnis
        data_dir = Path(__file__).parents[2] / "data" / "processed"
        demand_file = data_dir / "demand_germany_hourly.parquet"

        if demand_file.exists():
            try:
                df = pd.read_parquet(demand_file)

                # Sortiere Index und entferne Duplikate (Problem bei Zeitumstellung)
                df = df.sort_index()
                df = df[~df.index.duplicated(keep='first')]

                # Reindex auf unsere Snapshots (automatisches Interpolation/Alignment)
                demand = df['demand_mw'].reindex(self.snapshots, method='nearest', tolerance='1h')

                # Prüfe, ob genug Daten vorhanden sind
                if demand.notna().sum() / len(self.snapshots) > 0.9:  # Mindestens 90% Daten
                    self.logger.info(f"Loaded demand timeseries from {demand_file}")
                    return demand
                else:
                    self.logger.warning(f"Insufficient demand data coverage: {demand.notna().sum()}/{len(self.snapshots)}")

            except Exception as e:
                self.logger.warning(f"Failed to load demand timeseries: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

        return None

    def _generate_synthetic_demand(self) -> pd.Series:
        """
        Generiert synthetische Nachfrage-Zeitreihe.

        Basiert auf typischen Mustern:
        - Saisonale Variation
        - Tägliche Variation (höher am Tag, niedriger nachts)
        - Wochenmuster (Werktag vs. Wochenende)

        Returns:
        --------
        pd.Series
            Synthetische Nachfrage in MW
        """
        # Basis-Parameter aus Konfiguration
        annual_twh = self.config['demand'].get('annual_twh', DEFAULT_DEMAND['annual_twh'])
        peak_load_gw = self.config['demand'].get('peak_load_gw', DEFAULT_DEMAND['peak_load_gw'])

        # Durchschnittliche Last berechnen
        hours_per_year = 8760
        avg_load_gw = annual_twh / hours_per_year
        avg_load_mw = avg_load_gw * UNITS['gw_to_mw']
        peak_load_mw = peak_load_gw * UNITS['gw_to_mw']

        # Basis-Last
        demand = pd.Series(avg_load_mw, index=self.snapshots)

        # Tägliches Muster (24h-Zyklus)
        hours = self.snapshots.hour
        daily_pattern = 0.85 + 0.3 * np.sin((hours - 6) * np.pi / 12)
        daily_pattern = np.clip(daily_pattern, 0.7, 1.3)

        # Wochenmuster (Werktag vs. Wochenende)
        weekday = self.snapshots.dayofweek
        weekly_pattern = np.where(weekday < 5, 1.05, 0.9)  # Werktag höher

        # Saisonales Muster (Winter höher als Sommer)
        day_of_year = self.snapshots.dayofyear
        seasonal_pattern = 1.0 + 0.15 * np.cos((day_of_year - 1) * 2 * np.pi / 365)

        # Kombiniere alle Muster
        demand = demand * daily_pattern * weekly_pattern * seasonal_pattern

        # Zufälliges Rauschen hinzufügen
        np.random.seed(42)  # Für Reproduzierbarkeit
        noise = np.random.normal(1.0, 0.05, len(demand))
        demand = demand * noise

        # Stelle sicher, dass Peak nicht überschritten wird
        demand = np.clip(demand, 0.3 * peak_load_mw, peak_load_mw)

        return demand

    def _add_generators(self):
        """
        Fügt Kraftwerke hinzu.

        Alle Kraftwerke werden am Bus "DE" angeschlossen.
        Merit Order wird durch marginale Kosten definiert.
        """
        for tech, tech_data in GENERATOR_TECHNOLOGIES.items():
            # Kapazität aus Konfiguration
            gen_config = self.config['generators'].get(tech, {})

            # Handle both old format (float) and new format (dict)
            if isinstance(gen_config, dict):
                capacity_gw = gen_config.get('capacity_gw', 0.0)
                # Wenn marginal_cost im config ist, nutze es (für manuelle Overrides)
                # Ansonsten berechne mit CO2-Preis
                if 'marginal_cost' in gen_config:
                    marginal_cost = gen_config['marginal_cost']
                else:
                    marginal_cost = calculate_marginal_cost_with_co2(tech, self.co2_price)
            else:
                # Old format: just a float
                capacity_gw = gen_config
                # Berechne marginale Kosten inkl. CO2
                marginal_cost = calculate_marginal_cost_with_co2(tech, self.co2_price)

            if capacity_gw <= 0:
                continue

            capacity_mw = capacity_gw * UNITS['gw_to_mw']

            # Zeitreihe für variable Erzeugung (Wind, Solar)
            if tech in ['wind_onshore', 'wind_offshore', 'solar']:
                p_max_pu = self._get_renewable_profile(tech)
            else:
                # Dispatchable Kraftwerke: immer verfügbar
                p_max_pu = 1.0

            # Ramping-Constraints (falls vorhanden)
            ramp_limit_up = tech_data.get('ramp_limit_up', None)
            ramp_limit_down = tech_data.get('ramp_limit_down', None)

            # Must-Run Constraint (Mindestlast)
            p_min_pu = tech_data.get('p_min_pu', None)

            # Generator hinzufügen
            gen_kwargs = {
                "bus": "DE",
                "carrier": tech,
                "p_nom": capacity_mw,
                "marginal_cost": marginal_cost,
                "p_max_pu": p_max_pu
            }

            # Füge Ramping-Constraints nur hinzu, wenn sie definiert sind
            if ramp_limit_up is not None:
                gen_kwargs["ramp_limit_up"] = ramp_limit_up
            if ramp_limit_down is not None:
                gen_kwargs["ramp_limit_down"] = ramp_limit_down

            # Füge Must-Run Constraint hinzu
            if p_min_pu is not None:
                gen_kwargs["p_min_pu"] = p_min_pu

            self.network.add(
                "Generator",
                f"{tech}_DE",
                **gen_kwargs
            )

            ramp_info = ""
            if ramp_limit_up is not None:
                ramp_info = f", Ramp={ramp_limit_up*100:.0f}%/h"

            # Must-Run Info
            must_run_info = ""
            if p_min_pu is not None:
                must_run_info = f", MinLoad={p_min_pu*100:.0f}%"

            # Zeige CO2-Kosten-Anteil
            base_cost = tech_data['marginal_cost']
            co2_cost = marginal_cost - base_cost
            co2_info = ""
            if co2_cost > 0:
                co2_info = f" (Base: {base_cost:.1f} + CO2: {co2_cost:.1f})"

            self.logger.info(
                f"Added generator: {tech} ({capacity_mw:.0f} MW, "
                f"MC={marginal_cost:.1f} EUR/MWh{co2_info}{ramp_info}{must_run_info})"
            )

    def _get_renewable_profile(self, technology: str) -> pd.Series:
        """
        Lädt oder generiert Kapazitätsfaktor-Zeitreihe für Erneuerbare.

        Parameters:
        -----------
        technology : str
            Technologie (wind_onshore, wind_offshore, solar)

        Returns:
        --------
        pd.Series
            Kapazitätsfaktoren (0-1) für jeden Snapshot
        """
        # Versuche, vorhandene Profile zu laden
        profile = self._load_renewable_profile(technology)

        if profile is None:
            # Fallback: Generiere synthetische Profile
            self.logger.warning(
                f"No profile found for {technology}, generating synthetic profile"
            )
            profile = self._generate_synthetic_renewable_profile(technology)

        return profile

    def _load_renewable_profile(self, technology: str) -> Optional[pd.Series]:
        """Lädt historisches Erzeugungsprofil."""
        data_dir = Path(__file__).parents[2] / "data" / "processed"
        profile_file = data_dir / f"{technology}_germany_hourly.parquet"

        if profile_file.exists():
            try:
                df = pd.read_parquet(profile_file)

                # Sortiere Index und entferne Duplikate (Problem bei Zeitumstellung)
                df = df.sort_index()
                df = df[~df.index.duplicated(keep='first')]

                # Reindex auf unsere Snapshots
                profile = df['capacity_factor'].reindex(self.snapshots, method='nearest', tolerance='1h')

                # Prüfe Datenabdeckung
                if profile.notna().sum() / len(self.snapshots) > 0.9:
                    self.logger.info(f"Loaded {technology} profile from {profile_file}")
                    # Clip auf [0, 1] für Kapazitätsfaktoren
                    return profile.clip(0, 1)
                else:
                    self.logger.warning(
                        f"Insufficient {technology} data coverage: {profile.notna().sum()}/{len(self.snapshots)}"
                    )

            except Exception as e:
                self.logger.warning(f"Failed to load {technology} profile: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

        return None

    def _generate_synthetic_renewable_profile(self, technology: str) -> pd.Series:
        """Generiert synthetisches Erzeugungsprofil für Erneuerbare."""
        np.random.seed(42)

        if technology == 'solar':
            # Solar: Nur tagsüber, abhängig von Jahreszeit
            hours = self.snapshots.hour
            day_of_year = self.snapshots.dayofyear

            # Tagesmuster (nur zwischen 6-20 Uhr)
            daily = np.where(
                (hours >= 6) & (hours <= 20),
                np.sin((hours - 6) * np.pi / 14) ** 2,
                0.0
            )

            # Saisonales Muster (Sommer höher)
            seasonal = 0.3 + 0.7 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
            seasonal = np.clip(seasonal, 0.2, 1.0)

            profile = daily * seasonal

        elif technology in ['wind_onshore', 'wind_offshore']:
            # Wind: Mehr im Winter, zufälligere Variation
            day_of_year = self.snapshots.dayofyear

            # Saisonales Muster (Winter höher)
            seasonal = 0.5 + 0.3 * np.cos((day_of_year - 1) * 2 * np.pi / 365)

            # Zufälliges Muster mit Autokorrelation
            random_base = np.random.beta(2, 5, len(self.snapshots))

            # Glättung für Autokorrelation (Wind-Fronten dauern mehrere Stunden)
            window = 6
            profile = pd.Series(random_base).rolling(window=window, center=True).mean()
            profile = profile.fillna(method='bfill').fillna(method='ffill')

            profile = profile * seasonal

            # Offshore hat höhere Kapazitätsfaktoren
            if technology == 'wind_offshore':
                profile = profile * 1.3

        else:
            # Fallback: Konstant 50%
            profile = pd.Series(0.5, index=self.snapshots)

        # Normalisiere auf [0, 1]
        profile = np.clip(profile, 0.0, 1.0)

        return pd.Series(profile.values, index=self.snapshots)

    def _add_interconnectors(self):
        """
        Fügt virtuelle Import/Export-Kapazitäten hinzu.

        Import: Modelliert als Generator mit zeitvariablen marginalen Kosten
                (basierend auf echten Nachbarland-Preisen)
        Export: Modelliert als Last mit zeitvariablem Wert
                (basierend auf echten Nachbarland-Preisen)

        Verwendet gewichtete Import/Export-Preise aus SMARD-Daten.
        """
        # Versuche echte Import/Export-Preise zu laden
        data_dir = Path(__file__).parents[2] / "data" / "processed"
        weighted_prices_file = data_dir / "weighted_import_export_prices_hourly.parquet"

        use_real_prices = False
        import_prices = None
        export_prices = None

        if weighted_prices_file.exists():
            try:
                df_prices = pd.read_parquet(weighted_prices_file)

                # Schneide auf unsere Zeitreihe zu
                df_prices = df_prices.loc[self.snapshots]

                # Extrahiere Preise
                import_prices = df_prices['import_price_eur_per_mwh']
                export_prices = df_prices['export_price_eur_per_mwh']

                # Fülle fehlende Werte mit Durchschnitt
                import_prices = import_prices.fillna(import_prices.mean())
                export_prices = export_prices.fillna(export_prices.mean())

                use_real_prices = True
                self.logger.info(
                    f"Using real import/export prices from SMARD data:\n"
                    f"  Import: {import_prices.mean():.1f} ± {import_prices.std():.1f} EUR/MWh\n"
                    f"  Export: {export_prices.mean():.1f} ± {export_prices.std():.1f} EUR/MWh"
                )
            except Exception as e:
                self.logger.warning(f"Could not load real import/export prices: {e}")
                self.logger.warning("Falling back to fixed prices from config")

        # Import
        import_cap_gw = self.config['interconnectors']['import']['capacity_gw']

        if import_cap_gw > 0:
            if use_real_prices:
                # Verwende zeitvariable echte Preise
                marginal_cost = 0.0  # Basis-Kosten, wird überschrieben
            else:
                # Fallback auf feste Kosten
                marginal_cost = self.config['interconnectors']['import']['marginal_cost']

            self.network.add(
                "Generator",
                "import_DE",
                bus="DE",
                carrier="import",
                p_nom=import_cap_gw * UNITS['gw_to_mw'],
                marginal_cost=marginal_cost,
                p_max_pu=1.0
            )

            # Setze zeitvariable marginale Kosten
            if use_real_prices:
                self.network.generators_t.marginal_cost['import_DE'] = import_prices.values
                self.logger.info(
                    f"Added import: {import_cap_gw} GW with time-varying prices "
                    f"(avg: {import_prices.mean():.1f} EUR/MWh)"
                )
            else:
                self.logger.info(
                    f"Added import: {import_cap_gw} GW @ {marginal_cost} EUR/MWh (fixed)"
                )

        # Export (als Last mit positivem Nutzen)
        export_cap_gw = self.config['interconnectors']['export']['capacity_gw']

        if export_cap_gw > 0:
            if use_real_prices:
                # Export als Last, die Geld bringt
                # Negative Last = Export
                # Wir verwenden die Export-Preise als Wert für die exportierte Energie
                marginal_cost = 0.0  # Basis-Kosten
            else:
                marginal_cost = self.config['interconnectors']['export']['marginal_cost']

            # Export als Generator mit NEGATIVEN Grenzkosten
            # (d.h. System verdient beim Export)
            self.network.add(
                "Generator",
                "export_DE",
                bus="DE",
                carrier="export",
                p_nom=export_cap_gw * UNITS['gw_to_mw'],
                marginal_cost=marginal_cost,
                p_max_pu=1.0
            )

            # Setze zeitvariable marginale Kosten (negativ für Export)
            if use_real_prices:
                self.network.generators_t.marginal_cost['export_DE'] = -export_prices.values
                self.logger.info(
                    f"Added export: {export_cap_gw} GW with time-varying prices "
                    f"(avg: -{export_prices.mean():.1f} EUR/MWh)"
                )
            else:
                self.logger.info(
                    f"Added export: {export_cap_gw} GW @ {marginal_cost} EUR/MWh (fixed, negative MC = export)"
                )

    def _add_storage(self, storage_config: Dict):
        """
        Fügt Batteriespeicher hinzu.

        Parameters:
        -----------
        storage_config : dict
            Speicher-Konfiguration mit 'capacity_gwh' und 'power_gw'
        """
        capacity_gwh = storage_config.get('capacity_gwh', 0)
        power_gw = storage_config.get('power_gw', 0)

        if capacity_gwh <= 0 or power_gw <= 0:
            return

        capacity_mwh = capacity_gwh * UNITS['gw_to_mw']
        power_mw = power_gw * UNITS['gw_to_mw']

        battery_data = STORAGE_TECHNOLOGIES['battery']
        efficiency = battery_data['efficiency']
        marginal_cost = battery_data['marginal_cost']

        # Storage Unit hinzufügen
        self.network.add(
            "StorageUnit",
            "battery_DE",
            bus="DE",
            carrier="battery",
            p_nom=power_mw,  # Lade-/Entladeleistung
            max_hours=capacity_mwh / power_mw,  # Energiekapazität in Stunden
            efficiency_store=np.sqrt(efficiency),
            efficiency_dispatch=np.sqrt(efficiency),
            marginal_cost=marginal_cost,
            cyclic_state_of_charge=True  # State of Charge am Ende = am Anfang
        )

        self.logger.info(
            f"Added battery storage: {capacity_gwh} GWh, {power_gw} GW "
            f"(efficiency={efficiency:.2%})"
        )

    def _get_network_summary(self) -> str:
        """Gibt Zusammenfassung des Netzwerks zurück."""
        n_buses = len(self.network.buses)
        n_generators = len(self.network.generators)
        n_loads = len(self.network.loads)
        n_storage = len(self.network.storage_units)
        n_snapshots = len(self.network.snapshots)

        total_gen_capacity = self.network.generators.p_nom.sum() / 1000  # GW
        total_demand = self.network.loads_t.p_set.sum(axis=1).mean() / 1000  # GW

        return (
            f"{n_buses} buses, {n_generators} generators, {n_loads} loads, "
            f"{n_storage} storage units, {n_snapshots} snapshots, "
            f"{total_gen_capacity:.1f} GW generation capacity, "
            f"{total_demand:.1f} GW avg demand"
        )

    def save_network(self, filepath: str):
        """
        Speichert das Netzwerk auf die Festplatte.

        Parameters:
        -----------
        filepath : str
            Zielpfad (ohne Endung, wird automatisch als .nc gespeichert)
        """
        if self.network is None:
            raise ValueError("No network to save. Call build() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.network.export_to_netcdf(str(filepath))
        self.logger.info(f"Network saved to {filepath}")

    @staticmethod
    def load_network(filepath: str) -> pypsa.Network:
        """
        Lädt ein gespeichertes Netzwerk.

        Parameters:
        -----------
        filepath : str
            Pfad zur .nc Datei

        Returns:
        --------
        pypsa.Network
            Geladenes Netzwerk
        """
        logger = get_logger("network.builder")
        network = pypsa.Network(str(filepath))
        logger.info(f"Network loaded from {filepath}")
        return network


# Convenience function
def build_copper_plate_network(
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-31",
    scenario_config: Optional[Dict] = None,
    config_path: Optional[str] = None,
    co2_price: Optional[float] = None
) -> pypsa.Network:
    """
    Convenience-Funktion zum schnellen Erstellen eines Copper Plate Netzwerks.

    Parameters:
    -----------
    start_date : str
        Start-Datum (YYYY-MM-DD)
    end_date : str
        End-Datum (YYYY-MM-DD)
    scenario_config : dict, optional
        Szenario-spezifische Konfiguration
    config_path : str, optional
        Pfad zur Konfigurationsdatei
    co2_price : float, optional
        CO2-Preis in EUR/t. Falls None, wird CO2_PRICE['default'] verwendet.

    Returns:
    --------
    pypsa.Network
        Fertig konfiguriertes Netzwerk
    """
    builder = NetworkBuilder(
        config_path=config_path,
        start_date=start_date,
        end_date=end_date,
        co2_price=co2_price
    )

    return builder.build(scenario_config=scenario_config)


if __name__ == "__main__":
    # Test: Einfaches Netzwerk bauen
    from ..utils.logging_config import setup_logging

    setup_logging(level="INFO", console=True)

    # Baue Test-Netzwerk
    network = build_copper_plate_network(
        start_date="2024-01-01",
        end_date="2024-01-07",  # Nur 1 Woche für Test
        scenario_config={
            'storage': {
                'capacity_gwh': 10.0,
                'power_gw': 5.0
            }
        }
    )

    print("\n" + "="*60)
    print("Network Summary:")
    print("="*60)
    print(network)
