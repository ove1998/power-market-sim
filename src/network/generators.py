"""
Generator Management Module

Erweiterte Funktionen für Generator-Setup, Validierung und Konfiguration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import pypsa

from ..utils.logging_config import get_logger
from ..utils.constants import GENERATOR_TECHNOLOGIES, DEFAULT_CAPACITIES_GW, UNITS


logger = get_logger("network.generators")


def validate_generator_config(config: Dict) -> Tuple[bool, List[str]]:
    """
    Validiert Generator-Konfiguration.

    Parameters:
    -----------
    config : dict
        Generator-Konfiguration mit Kapazitäten pro Technologie

    Returns:
    --------
    bool
        True wenn valide
    List[str]
        Liste von Fehlermeldungen (leer wenn valide)
    """
    errors = []

    for tech, capacity in config.items():
        # Prüfe ob Technologie bekannt ist
        if tech not in GENERATOR_TECHNOLOGIES:
            errors.append(f"Unknown generator technology: {tech}")
            continue

        # Prüfe ob Kapazität positiv ist
        if capacity < 0:
            errors.append(f"Negative capacity for {tech}: {capacity}")

        # Warnung bei unrealistisch hohen Werten
        if capacity > 500:  # GW
            logger.warning(f"Very high capacity for {tech}: {capacity} GW")

    return len(errors) == 0, errors


def get_generation_mix_summary(network: pypsa.Network) -> pd.DataFrame:
    """
    Gibt Zusammenfassung des Kraftwerksparks zurück.

    Parameters:
    -----------
    network : pypsa.Network
        PyPSA-Netzwerk

    Returns:
    --------
    pd.DataFrame
        Tabelle mit Kapazitäten, Kosten und Labels pro Technologie
    """
    if len(network.generators) == 0:
        return pd.DataFrame()

    # Extrahiere Generator-Daten
    gen_data = []

    for idx, gen in network.generators.iterrows():
        carrier = gen.get('carrier', 'unknown')

        # Technologie-Daten
        tech_data = GENERATOR_TECHNOLOGIES.get(carrier, {})

        gen_data.append({
            'name': idx,
            'carrier': carrier,
            'label': tech_data.get('label', carrier),
            'capacity_mw': gen['p_nom'],
            'capacity_gw': gen['p_nom'] * UNITS['mw_to_gw'],
            'marginal_cost': gen.get('marginal_cost', 0.0),
            'color': tech_data.get('color', '#CCCCCC')
        })

    df = pd.DataFrame(gen_data)

    # Sortiere nach marginal cost (Merit Order)
    df = df.sort_values('marginal_cost')

    return df


def calculate_total_capacity_by_carrier(network: pypsa.Network) -> pd.Series:
    """
    Berechnet Gesamtkapazität pro Carrier/Technologie.

    Parameters:
    -----------
    network : pypsa.Network
        PyPSA-Netzwerk

    Returns:
    --------
    pd.Series
        Kapazität in GW pro Carrier
    """
    if len(network.generators) == 0:
        return pd.Series(dtype=float)

    capacities = network.generators.groupby('carrier')['p_nom'].sum()
    capacities = capacities * UNITS['mw_to_gw']

    return capacities.sort_values(ascending=False)


def get_merit_order_curve(network: pypsa.Network) -> pd.DataFrame:
    """
    Erstellt Merit Order Kurve aus dem Kraftwerkspark.

    Parameters:
    -----------
    network : pypsa.Network
        PyPSA-Netzwerk

    Returns:
    --------
    pd.DataFrame
        Tabelle mit kumulativer Kapazität und marginalen Kosten
    """
    if len(network.generators) == 0:
        return pd.DataFrame()

    # Sortiere Generatoren nach marginalen Kosten
    gens = network.generators.sort_values('marginal_cost')

    # Berechne kumulative Kapazität
    merit_order = []
    cumulative_capacity = 0.0

    for idx, gen in gens.iterrows():
        capacity_gw = gen['p_nom'] * UNITS['mw_to_gw']
        marginal_cost = gen.get('marginal_cost', 0.0)
        carrier = gen.get('carrier', 'unknown')

        merit_order.append({
            'generator': idx,
            'carrier': carrier,
            'capacity_gw': capacity_gw,
            'cumulative_capacity_gw': cumulative_capacity + capacity_gw,
            'marginal_cost': marginal_cost
        })

        cumulative_capacity += capacity_gw

    return pd.DataFrame(merit_order)


def estimate_available_capacity(
    network: pypsa.Network,
    snapshot: pd.Timestamp
) -> pd.DataFrame:
    """
    Schätzt verfügbare Kapazität zu einem bestimmten Zeitpunkt.

    Berücksichtigt p_max_pu für erneuerbare Energien.

    Parameters:
    -----------
    network : pypsa.Network
        PyPSA-Netzwerk
    snapshot : pd.Timestamp
        Zeitpunkt

    Returns:
    --------
    pd.DataFrame
        Verfügbare Kapazität pro Generator
    """
    if len(network.generators) == 0:
        return pd.DataFrame()

    available = []

    for idx, gen in network.generators.iterrows():
        p_nom = gen['p_nom']

        # Hole p_max_pu für den Zeitpunkt
        if idx in network.generators_t.p_max_pu.columns:
            p_max_pu = network.generators_t.p_max_pu.loc[snapshot, idx]
        else:
            p_max_pu = gen.get('p_max_pu', 1.0)

        available_mw = p_nom * p_max_pu
        available_gw = available_mw * UNITS['mw_to_gw']

        available.append({
            'generator': idx,
            'carrier': gen.get('carrier', 'unknown'),
            'p_nom_gw': p_nom * UNITS['mw_to_gw'],
            'p_max_pu': p_max_pu,
            'available_gw': available_gw,
            'marginal_cost': gen.get('marginal_cost', 0.0)
        })

    df = pd.DataFrame(available)
    df = df.sort_values('marginal_cost')

    return df


def add_generator_from_config(
    network: pypsa.Network,
    name: str,
    carrier: str,
    bus: str,
    capacity_gw: float,
    marginal_cost: Optional[float] = None,
    p_max_pu: Optional[pd.Series] = None,
    **kwargs
) -> None:
    """
    Fügt Generator aus Konfiguration hinzu (Helper-Funktion).

    Parameters:
    -----------
    network : pypsa.Network
        Ziel-Netzwerk
    name : str
        Generator-Name
    carrier : str
        Technologie/Carrier
    bus : str
        Angeschlossener Bus
    capacity_gw : float
        Kapazität in GW
    marginal_cost : float, optional
        Marginale Kosten (wird aus GENERATOR_TECHNOLOGIES geladen wenn None)
    p_max_pu : pd.Series, optional
        Verfügbarkeitsfaktor-Zeitreihe
    **kwargs
        Weitere Parameter für network.add()
    """
    # Hole Technologie-Daten
    tech_data = GENERATOR_TECHNOLOGIES.get(carrier, {})

    if marginal_cost is None:
        marginal_cost = tech_data.get('marginal_cost', 0.0)

    capacity_mw = capacity_gw * UNITS['gw_to_mw']

    if p_max_pu is None:
        p_max_pu = 1.0

    network.add(
        "Generator",
        name,
        bus=bus,
        carrier=carrier,
        p_nom=capacity_mw,
        marginal_cost=marginal_cost,
        p_max_pu=p_max_pu,
        **kwargs
    )

    logger.info(
        f"Added generator: {name} ({carrier}, {capacity_gw:.1f} GW, "
        f"MC={marginal_cost:.1f} EUR/MWh)"
    )


def scale_generators(
    network: pypsa.Network,
    scaling_factors: Dict[str, float]
) -> None:
    """
    Skaliert Generator-Kapazitäten.

    Nützlich für Sensitivitätsanalysen.

    Parameters:
    -----------
    network : pypsa.Network
        PyPSA-Netzwerk (wird in-place modifiziert)
    scaling_factors : dict
        Skalierungsfaktoren pro Carrier (z.B. {'wind_onshore': 1.5})
    """
    for carrier, factor in scaling_factors.items():
        # Finde alle Generatoren mit diesem Carrier
        mask = network.generators.carrier == carrier

        if mask.sum() == 0:
            logger.warning(f"No generators found with carrier '{carrier}'")
            continue

        # Skaliere p_nom
        old_capacity = network.generators.loc[mask, 'p_nom'].sum()
        network.generators.loc[mask, 'p_nom'] *= factor
        new_capacity = network.generators.loc[mask, 'p_nom'].sum()

        logger.info(
            f"Scaled {carrier}: {old_capacity * UNITS['mw_to_gw']:.1f} GW → "
            f"{new_capacity * UNITS['mw_to_gw']:.1f} GW (factor={factor:.2f})"
        )


def get_renewable_share(
    network: pypsa.Network,
    renewables: Optional[List[str]] = None
) -> float:
    """
    Berechnet Anteil erneuerbarer Energien an installierter Kapazität.

    Parameters:
    -----------
    network : pypsa.Network
        PyPSA-Netzwerk
    renewables : list of str, optional
        Liste von Carrier-Namen für Erneuerbare. Wenn None, wird Standard-Liste verwendet.

    Returns:
    --------
    float
        Anteil erneuerbarer Energien (0-1)
    """
    if renewables is None:
        renewables = [
            'wind_onshore',
            'wind_offshore',
            'solar',
            'hydro_run_of_river',
            'hydro_reservoir'
        ]

    if len(network.generators) == 0:
        return 0.0

    total_capacity = network.generators.p_nom.sum()

    if total_capacity == 0:
        return 0.0

    renewable_capacity = network.generators[
        network.generators.carrier.isin(renewables)
    ].p_nom.sum()

    return renewable_capacity / total_capacity


def get_generator_statistics(network: pypsa.Network) -> Dict:
    """
    Berechnet diverse Statistiken über den Kraftwerkspark.

    Parameters:
    -----------
    network : pypsa.Network
        PyPSA-Netzwerk

    Returns:
    --------
    dict
        Dictionary mit Statistiken
    """
    if len(network.generators) == 0:
        return {
            'n_generators': 0,
            'total_capacity_gw': 0.0,
            'renewable_share': 0.0,
            'n_carriers': 0
        }

    total_capacity_gw = network.generators.p_nom.sum() * UNITS['mw_to_gw']
    renewable_share = get_renewable_share(network)
    n_carriers = network.generators.carrier.nunique()

    # Durchschnittliche marginale Kosten (gewichtet nach Kapazität)
    weighted_mc = (
        network.generators.p_nom * network.generators.marginal_cost
    ).sum() / network.generators.p_nom.sum()

    # Kapazität pro Carrier
    capacity_by_carrier = calculate_total_capacity_by_carrier(network)

    return {
        'n_generators': len(network.generators),
        'total_capacity_gw': total_capacity_gw,
        'renewable_share': renewable_share,
        'n_carriers': n_carriers,
        'weighted_marginal_cost': weighted_mc,
        'capacity_by_carrier': capacity_by_carrier.to_dict()
    }


def create_generator_color_map(network: pypsa.Network) -> Dict[str, str]:
    """
    Erstellt Color-Map für Generatoren basierend auf Carrier.

    Parameters:
    -----------
    network : pypsa.Network
        PyPSA-Netzwerk

    Returns:
    --------
    dict
        Generator-Name → Farbe (Hex)
    """
    color_map = {}

    for idx, gen in network.generators.iterrows():
        carrier = gen.get('carrier', 'unknown')
        tech_data = GENERATOR_TECHNOLOGIES.get(carrier, {})
        color = tech_data.get('color', '#CCCCCC')
        color_map[idx] = color

    return color_map


def export_generators_to_csv(network: pypsa.Network, filepath: str) -> None:
    """
    Exportiert Generator-Konfiguration als CSV.

    Parameters:
    -----------
    network : pypsa.Network
        PyPSA-Netzwerk
    filepath : str
        Ziel-Dateipfad
    """
    if len(network.generators) == 0:
        logger.warning("No generators to export")
        return

    # Bereite Export-Daten vor
    export_df = network.generators.copy()
    export_df['capacity_gw'] = export_df['p_nom'] * UNITS['mw_to_gw']

    # Füge Technologie-Labels hinzu
    export_df['label'] = export_df['carrier'].apply(
        lambda c: GENERATOR_TECHNOLOGIES.get(c, {}).get('label', c)
    )

    # Sortiere nach marginal cost
    export_df = export_df.sort_values('marginal_cost')

    # Speichere
    export_df.to_csv(filepath, index=True, encoding='utf-8')
    logger.info(f"Exported {len(export_df)} generators to {filepath}")


# Test-Funktion
if __name__ == "__main__":
    # Test-Validierung
    test_config = {
        'nuclear': 8.0,
        'wind_onshore': 60.0,
        'solar': 70.0,
        'invalid_tech': 10.0,  # Sollte Fehler erzeugen
        'ccgt': -5.0  # Sollte Fehler erzeugen
    }

    valid, errors = validate_generator_config(test_config)

    print("Validation Test:")
    print(f"Valid: {valid}")
    print(f"Errors: {errors}")

    # Test mit echtem Netzwerk würde hier folgen
    # (benötigt ein gebautes Netzwerk)
