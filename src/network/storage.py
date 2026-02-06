"""
Storage Management Module

Funktionen für Batteriespeicher-Setup, Validierung und Analyse.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import pypsa

from ..utils.logging_config import get_logger
from ..utils.constants import STORAGE_TECHNOLOGIES, STORAGE_DISTRIBUTIONS, UNITS


logger = get_logger("network.storage")


def validate_storage_config(config: Dict) -> Tuple[bool, List[str]]:
    """
    Validiert Speicher-Konfiguration.

    Parameters:
    -----------
    config : dict
        Speicher-Konfiguration mit 'capacity_gwh' und 'power_gw'

    Returns:
    --------
    bool
        True wenn valide
    List[str]
        Liste von Fehlermeldungen
    """
    errors = []

    # Kapazität prüfen
    if 'capacity_gwh' not in config:
        errors.append("Missing 'capacity_gwh' in storage config")
    elif config['capacity_gwh'] < 0:
        errors.append(f"Negative capacity: {config['capacity_gwh']}")

    # Leistung prüfen
    if 'power_gw' not in config:
        errors.append("Missing 'power_gw' in storage config")
    elif config['power_gw'] < 0:
        errors.append(f"Negative power: {config['power_gw']}")

    # C-Rate prüfen (Verhältnis Leistung zu Kapazität)
    if 'capacity_gwh' in config and 'power_gw' in config:
        if config['capacity_gwh'] > 0:
            c_rate = config['power_gw'] / config['capacity_gwh']

            # Typische Batterien: C-Rate zwischen 0.25 und 2.0
            if c_rate < 0.1:
                logger.warning(f"Very low C-rate: {c_rate:.2f} (power-limited)")
            elif c_rate > 4.0:
                logger.warning(f"Very high C-rate: {c_rate:.2f} (energy-limited)")

    return len(errors) == 0, errors


def add_storage_from_config(
    network: pypsa.Network,
    name: str,
    bus: str,
    capacity_gwh: float,
    power_gw: float,
    efficiency: Optional[float] = None,
    marginal_cost: Optional[float] = None,
    cyclic: bool = True,
    **kwargs
) -> None:
    """
    Fügt Batteriespeicher zum Netzwerk hinzu.

    Parameters:
    -----------
    network : pypsa.Network
        Ziel-Netzwerk
    name : str
        Speicher-Name
    bus : str
        Angeschlossener Bus
    capacity_gwh : float
        Energiekapazität in GWh
    power_gw : float
        Lade-/Entladeleistung in GW
    efficiency : float, optional
        Round-trip Effizienz (wenn None, wird aus STORAGE_TECHNOLOGIES geladen)
    marginal_cost : float, optional
        Betriebskosten EUR/MWh
    cyclic : bool
        Ob State of Charge am Ende = am Anfang
    **kwargs
        Weitere Parameter
    """
    # Lade Standard-Werte
    battery_data = STORAGE_TECHNOLOGIES.get('battery', {})

    if efficiency is None:
        efficiency = battery_data.get('efficiency', 0.90)

    if marginal_cost is None:
        marginal_cost = battery_data.get('marginal_cost', 0.1)

    # Konvertiere Einheiten
    capacity_mwh = capacity_gwh * UNITS['gw_to_mw']
    power_mw = power_gw * UNITS['gw_to_mw']

    # Berechne max_hours
    max_hours = capacity_mwh / power_mw if power_mw > 0 else 0

    # Speicher hinzufügen
    network.add(
        "StorageUnit",
        name,
        bus=bus,
        carrier="battery",
        p_nom=power_mw,
        max_hours=max_hours,
        efficiency_store=np.sqrt(efficiency),  # Lade-Effizienz
        efficiency_dispatch=np.sqrt(efficiency),  # Entlade-Effizienz
        marginal_cost=marginal_cost,
        cyclic_state_of_charge=cyclic,
        **kwargs
    )

    c_rate = power_gw / capacity_gwh if capacity_gwh > 0 else 0

    logger.info(
        f"Added storage: {name} ({capacity_gwh:.1f} GWh, {power_gw:.1f} GW, "
        f"C-rate={c_rate:.2f}, efficiency={efficiency:.1%})"
    )


def calculate_storage_statistics(network: pypsa.Network) -> Dict:
    """
    Berechnet Statistiken über Speicher im Netzwerk.

    Parameters:
    -----------
    network : pypsa.Network
        PyPSA-Netzwerk

    Returns:
    --------
    dict
        Dictionary mit Statistiken
    """
    if len(network.storage_units) == 0:
        return {
            'n_storage': 0,
            'total_power_gw': 0.0,
            'total_capacity_gwh': 0.0,
            'avg_c_rate': 0.0,
            'avg_efficiency': 0.0
        }

    # Gesamtleistung
    total_power_gw = network.storage_units.p_nom.sum() * UNITS['mw_to_gw']

    # Gesamtkapazität (Energy)
    total_capacity_gwh = (
        network.storage_units.p_nom * network.storage_units.max_hours
    ).sum() * UNITS['mw_to_gw']

    # Durchschnittliche C-Rate
    if total_capacity_gwh > 0:
        avg_c_rate = total_power_gw / total_capacity_gwh
    else:
        avg_c_rate = 0.0

    # Durchschnittliche Effizienz (Round-trip)
    avg_efficiency_store = network.storage_units.efficiency_store.mean()
    avg_efficiency_dispatch = network.storage_units.efficiency_dispatch.mean()
    avg_efficiency_roundtrip = avg_efficiency_store * avg_efficiency_dispatch

    return {
        'n_storage': len(network.storage_units),
        'total_power_gw': total_power_gw,
        'total_capacity_gwh': total_capacity_gwh,
        'avg_c_rate': avg_c_rate,
        'avg_efficiency': avg_efficiency_roundtrip
    }


def get_storage_timeseries(
    network: pypsa.Network,
    storage_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Extrahiert Zeitreihen für Speicher (nach Optimierung).

    Parameters:
    -----------
    network : pypsa.Network
        Optimiertes PyPSA-Netzwerk
    storage_name : str, optional
        Name eines spezifischen Speichers. Wenn None, werden alle Speicher zurückgegeben.

    Returns:
    --------
    pd.DataFrame
        Zeitreihen mit p (Leistung) und state_of_charge
    """
    if len(network.storage_units) == 0:
        return pd.DataFrame()

    # Prüfe ob Netzwerk optimiert wurde
    if not hasattr(network.storage_units_t, 'p') or network.storage_units_t.p.empty:
        logger.warning("Network not optimized yet - no timeseries available")
        return pd.DataFrame()

    if storage_name:
        # Einzelner Speicher
        if storage_name not in network.storage_units.index:
            logger.error(f"Storage '{storage_name}' not found")
            return pd.DataFrame()

        df = pd.DataFrame({
            'p_mw': network.storage_units_t.p[storage_name],
            'state_of_charge_mwh': network.storage_units_t.state_of_charge[storage_name]
        })

        # Konvertiere in GW/GWh
        df['p_gw'] = df['p_mw'] * UNITS['mw_to_gw']
        df['state_of_charge_gwh'] = df['state_of_charge_mwh'] * UNITS['mw_to_gw']

    else:
        # Alle Speicher
        df = pd.DataFrame({
            'p_mw': network.storage_units_t.p.sum(axis=1),
            'state_of_charge_mwh': network.storage_units_t.state_of_charge.sum(axis=1)
        })

        df['p_gw'] = df['p_mw'] * UNITS['mw_to_gw']
        df['state_of_charge_gwh'] = df['state_of_charge_mwh'] * UNITS['mw_to_gw']

    return df


def calculate_storage_cycles(network: pypsa.Network, storage_name: str) -> float:
    """
    Berechnet Vollzyklus-Äquivalent für einen Speicher.

    Ein Vollzyklus = Entladung der gesamten Kapazität.

    Parameters:
    -----------
    network : pypsa.Network
        Optimiertes Netzwerk
    storage_name : str
        Name des Speichers

    Returns:
    --------
    float
        Anzahl Vollzyklen
    """
    if storage_name not in network.storage_units.index:
        logger.error(f"Storage '{storage_name}' not found")
        return 0.0

    if not hasattr(network.storage_units_t, 'p') or network.storage_units_t.p.empty:
        logger.warning("Network not optimized yet")
        return 0.0

    # Kapazität
    capacity_mwh = (
        network.storage_units.at[storage_name, 'p_nom'] *
        network.storage_units.at[storage_name, 'max_hours']
    )

    if capacity_mwh == 0:
        return 0.0

    # Summiere alle Entladevorgänge (positive Leistung)
    discharge = network.storage_units_t.p[storage_name]
    total_discharge_mwh = discharge[discharge > 0].sum()

    # Anzahl Vollzyklen
    cycles = total_discharge_mwh / capacity_mwh

    return cycles


def calculate_storage_revenue(
    network: pypsa.Network,
    storage_name: str,
    electricity_prices: pd.Series
) -> Dict:
    """
    Berechnet Erlöse eines Speichers basierend auf Strompreisen.

    Parameters:
    -----------
    network : pypsa.Network
        Optimiertes Netzwerk
    storage_name : str
        Name des Speichers
    electricity_prices : pd.Series
        Strompreise in EUR/MWh

    Returns:
    --------
    dict
        Revenue-Metriken
    """
    if storage_name not in network.storage_units.index:
        logger.error(f"Storage '{storage_name}' not found")
        return {}

    if not hasattr(network.storage_units_t, 'p') or network.storage_units_t.p.empty:
        logger.warning("Network not optimized yet")
        return {}

    # Leistungs-Zeitreihe (positiv = Entladung, negativ = Ladung)
    p_mw = network.storage_units_t.p[storage_name]

    # Erlöse = Entladung * Preis, Kosten = Ladung * Preis
    revenue = (p_mw * electricity_prices).sum()  # EUR

    # Separate Betrachtung
    discharge = p_mw[p_mw > 0]
    charge = p_mw[p_mw < 0].abs()

    revenue_from_discharge = (discharge * electricity_prices[discharge.index]).sum()
    cost_from_charge = (charge * electricity_prices[charge.index]).sum()

    # Kapazitäten
    capacity_mwh = (
        network.storage_units.at[storage_name, 'p_nom'] *
        network.storage_units.at[storage_name, 'max_hours']
    )
    capacity_gwh = capacity_mwh * UNITS['mw_to_gw']

    # Vollzyklen
    cycles = calculate_storage_cycles(network, storage_name)

    # Revenue pro GWh Kapazität
    if capacity_gwh > 0:
        revenue_per_gwh = revenue / capacity_gwh
    else:
        revenue_per_gwh = 0.0

    return {
        'total_revenue': revenue,
        'revenue_from_discharge': revenue_from_discharge,
        'cost_from_charge': cost_from_charge,
        'net_revenue': revenue_from_discharge - cost_from_charge,
        'capacity_gwh': capacity_gwh,
        'revenue_per_gwh': revenue_per_gwh,
        'cycles': cycles
    }


def distribute_storage_capacity(
    total_capacity_gwh: float,
    total_power_gw: float,
    n_locations: int,
    distribution_type: str = 'uniform',
    weights: Optional[List[float]] = None
) -> List[Dict]:
    """
    Verteilt Speicherkapazität auf mehrere Standorte.

    Nützlich für geografische Verteilungs-Szenarien.

    Parameters:
    -----------
    total_capacity_gwh : float
        Gesamte Energiekapazität
    total_power_gw : float
        Gesamte Leistung
    n_locations : int
        Anzahl Standorte
    distribution_type : str
        'uniform', 'north_heavy', 'south_heavy', 'demand_proportional'
    weights : list of float, optional
        Custom Gewichte für Verteilung (Summe sollte 1.0 sein)

    Returns:
    --------
    list of dict
        Liste mit Speicher-Konfigurationen pro Standort
    """
    if n_locations <= 0:
        return []

    if weights is None:
        if distribution_type == 'uniform':
            weights = [1.0 / n_locations] * n_locations

        elif distribution_type == 'north_heavy':
            # Mehr im Norden (für Wind)
            weights = np.linspace(1.5, 0.5, n_locations)
            weights = weights / weights.sum()

        elif distribution_type == 'south_heavy':
            # Mehr im Süden (für Solar)
            weights = np.linspace(0.5, 1.5, n_locations)
            weights = weights / weights.sum()

        elif distribution_type == 'demand_proportional':
            # Vereinfacht: Gleichverteilt (ohne echte Demand-Daten)
            logger.warning(
                "demand_proportional requires actual demand data, "
                "falling back to uniform"
            )
            weights = [1.0 / n_locations] * n_locations

        else:
            logger.warning(f"Unknown distribution type '{distribution_type}', using uniform")
            weights = [1.0 / n_locations] * n_locations

    # Normalisiere Gewichte
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Verteile Kapazitäten
    storage_configs = []

    for i, weight in enumerate(weights):
        config = {
            'location_id': i,
            'capacity_gwh': total_capacity_gwh * weight,
            'power_gw': total_power_gw * weight,
            'weight': weight
        }
        storage_configs.append(config)

    logger.info(
        f"Distributed {total_capacity_gwh:.1f} GWh across {n_locations} locations "
        f"using '{distribution_type}' distribution"
    )

    return storage_configs


def get_storage_color_map() -> Dict[str, str]:
    """
    Gibt Color-Map für Speicher-Visualisierung zurück.

    Returns:
    --------
    dict
        Carrier → Color
    """
    color_map = {}

    for storage_type, data in STORAGE_TECHNOLOGIES.items():
        color_map[storage_type] = data.get('color', '#9B59B6')

    return color_map


def export_storage_to_csv(network: pypsa.Network, filepath: str) -> None:
    """
    Exportiert Speicher-Konfiguration als CSV.

    Parameters:
    -----------
    network : pypsa.Network
        PyPSA-Netzwerk
    filepath : str
        Ziel-Dateipfad
    """
    if len(network.storage_units) == 0:
        logger.warning("No storage units to export")
        return

    # Bereite Export-Daten vor
    export_df = network.storage_units.copy()

    # Konvertiere Einheiten
    export_df['power_gw'] = export_df['p_nom'] * UNITS['mw_to_gw']
    export_df['capacity_gwh'] = (
        export_df['p_nom'] * export_df['max_hours'] * UNITS['mw_to_gw']
    )
    export_df['c_rate'] = export_df['power_gw'] / export_df['capacity_gwh']
    export_df['efficiency_roundtrip'] = (
        export_df['efficiency_store'] * export_df['efficiency_dispatch']
    )

    # Speichere
    export_df.to_csv(filepath, index=True, encoding='utf-8')
    logger.info(f"Exported {len(export_df)} storage units to {filepath}")


# Test-Funktion
if __name__ == "__main__":
    # Test-Validierung
    test_config = {
        'capacity_gwh': 10.0,
        'power_gw': 5.0
    }

    valid, errors = validate_storage_config(test_config)

    print("Validation Test:")
    print(f"Valid: {valid}")
    print(f"Errors: {errors}")

    # Test Verteilung
    print("\nDistribution Test:")
    configs = distribute_storage_capacity(
        total_capacity_gwh=100.0,
        total_power_gw=50.0,
        n_locations=4,
        distribution_type='north_heavy'
    )

    for config in configs:
        print(
            f"Location {config['location_id']}: "
            f"{config['capacity_gwh']:.1f} GWh, "
            f"{config['power_gw']:.1f} GW "
            f"(weight={config['weight']:.2%})"
        )
