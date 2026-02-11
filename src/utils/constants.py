"""
Constants for Power Market Simulation

Enthält wichtige Konstanten für Technologien, Kosten und Konfigurationen.
"""

# CO2 Emission Factors (kg CO2 per MWh)
# Sources: IPCC, UBA Deutschland
CO2_EMISSION_FACTORS = {
    'nuclear': 0.0,      # Keine direkten CO2-Emissionen
    'lignite': 1100.0,   # Braunkohle: ~1.1 t CO2/MWh
    'hard_coal': 900.0,  # Steinkohle: ~0.9 t CO2/MWh
    'ccgt': 400.0,       # Gas GuD: ~0.4 t CO2/MWh
    'ocgt': 450.0,       # Gas GT: ~0.45 t CO2/MWh (weniger effizient)
    'wind_onshore': 0.0,
    'wind_offshore': 0.0,
    'solar': 0.0,
    'hydro_run_of_river': 0.0,
}

# CO2 Price Parameters
CO2_PRICE = {
    'default': 75.0,  # EUR per tonne CO2 (EU ETS 2024-2025)
    'min': 30.0,      # Historisches Minimum
    'max': 100.0,     # Mögliches Maximum
}

# Generator Technologies
GENERATOR_TECHNOLOGIES = {
    'nuclear': {
        'marginal_cost': 5.0,  # EUR/MWh (Betriebskosten, ohne Brennstoff)
        'fuel_cost': 5.0,      # EUR/MWh Brennstoffkosten
        'co2_factor': CO2_EMISSION_FACTORS['nuclear'],
        'color': '#FF6B6B',
        'label': 'Kernkraft',
        'ramp_limit_up': 0.05,  # 5% pro Stunde (relativ zu p_nom)
        'ramp_limit_down': 0.05,
        # Kein p_min_pu - Nuclear läuft meist durch, wird durch niedrige MC modelliert
    },
    'lignite': {
        'marginal_cost': 25.0,  # EUR/MWh (Betriebskosten + Brennstoff, erhöht für 2024-2025)
        'fuel_cost': 20.0,      # EUR/MWh Brennstoffkosten (erhöht)
        'co2_factor': CO2_EMISSION_FACTORS['lignite'],
        'color': '#8B4513',
        'label': 'Braunkohle',
        'ramp_limit_up': 0.10,  # 10% pro Stunde - Braunkohle sehr träge
        'ramp_limit_down': 0.10,
        # Kein p_min_pu - in linearem Modell problematisch
    },
    'hard_coal': {
        'marginal_cost': 50.0,  # EUR/MWh (Betriebskosten + Brennstoff, erhöht für 2024-2025)
        'fuel_cost': 40.0,      # EUR/MWh Brennstoffkosten (erhöht)
        'co2_factor': CO2_EMISSION_FACTORS['hard_coal'],
        'color': '#2F4F4F',
        'label': 'Steinkohle',
        'ramp_limit_up': 0.15,  # 15% pro Stunde - etwas flexibler als Braunkohle
        'ramp_limit_down': 0.15,
        # Kein p_min_pu
    },
    'ccgt': {
        'marginal_cost': 115.0,  # EUR/MWh (Betriebskosten + Brennstoff, 2024-2025 Level)
        'fuel_cost': 105.0,      # EUR/MWh Brennstoffkosten (erhöht für 2024-2025 Gaspreise)
        'co2_factor': CO2_EMISSION_FACTORS['ccgt'],
        'color': '#4ECDC4',
        'label': 'Gas GuD',
        'ramp_limit_up': 0.30,  # 30% pro Stunde - Gas relativ flexibel
        'ramp_limit_down': 0.30,
        # Kein p_min_pu
    },
    'ocgt': {
        'marginal_cost': 140.0,  # EUR/MWh (Betriebskosten + Brennstoff, 2024-2025 Level)
        'fuel_cost': 125.0,      # EUR/MWh Brennstoffkosten (erhöht für 2024-2025 Gaspreise)
        'co2_factor': CO2_EMISSION_FACTORS['ocgt'],
        'color': '#95E1D3',
        'label': 'Gas GT',
        'ramp_limit_up': 1.0,  # 100% pro Stunde - Gasturbinen sehr flexibel
        'ramp_limit_down': 1.0,
        # Kein p_min_pu
    },
    'wind_onshore': {
        'marginal_cost': 0.0,  # Grenzkosten im Merit Order Dispatch
        'fuel_cost': 0.0,
        'co2_factor': CO2_EMISSION_FACTORS['wind_onshore'],
        'color': '#1ABC9C',
        'label': 'Wind Onshore',
        # Keine Ramping-Constraints - wetterabhängig
    },
    'wind_offshore': {
        'marginal_cost': 0.0,  # Grenzkosten im Merit Order Dispatch
        'fuel_cost': 0.0,
        'co2_factor': CO2_EMISSION_FACTORS['wind_offshore'],
        'color': '#16A085',
        'label': 'Wind Offshore',
        # Keine Ramping-Constraints - wetterabhängig
    },
    'solar': {
        'marginal_cost': 0.0,  # Grenzkosten im Merit Order Dispatch
        'fuel_cost': 0.0,
        'co2_factor': CO2_EMISSION_FACTORS['solar'],
        'color': '#F39C12',
        'label': 'Solar PV',
        # Keine Ramping-Constraints - wetterabhängig
    },
    'hydro_run_of_river': {
        'marginal_cost': 0.0,
        'fuel_cost': 0.0,
        'co2_factor': CO2_EMISSION_FACTORS['hydro_run_of_river'],
        'color': '#3498DB',
        'label': 'Wasserkraft (Laufwasser)',
        # Keine Ramping-Constraints - flussabhängig
    },
}

# Storage Technologies
STORAGE_TECHNOLOGIES = {
    'battery': {
        'efficiency': 0.90,  # Round-trip efficiency
        'marginal_cost': 0.1,  # EUR/MWh (Betriebskosten)
        'color': '#9B59B6',
        'label': 'Batteriespeicher'
    },
    'pumped_hydro': {
        'efficiency': 0.78,  # Round-trip efficiency Pumpspeicher
        'marginal_cost': 1.0,  # EUR/MWh (Betriebskosten)
        'color': '#2980B9',
        'label': 'Pumpspeicher'
    },
}

# Default Capacities (Deutschland, Stand 2024-2025, aus SMARD-Daten)
DEFAULT_CAPACITIES_GW = {
    'nuclear': 8.0,
    'lignite': 20.0,
    'hard_coal': 25.0,
    'ccgt': 30.0,
    'ocgt': 5.0,
    'wind_onshore': 46.0,   # Reduziert von 60 auf echte SMARD Max-Kapazität
    'wind_offshore': 8.0,    # Reduziert von 10 auf echte SMARD Max-Kapazität
    'solar': 52.0,           # Reduziert von 70 auf echte SMARD Max-Kapazität
    'hydro_run_of_river': 4.0,
}

# Demand Parameters
DEFAULT_DEMAND = {
    'annual_twh': 500.0,  # Deutschland Jahresverbrauch
    'peak_load_gw': 80.0,  # Spitzenlast
}

# Import/Export Parameters
INTERCONNECTOR_PARAMS = {
    'import': {
        'capacity_gw': 15.0,
        'marginal_cost': 150.0,  # EUR/MWh - aktiviert nur bei sehr hohen Preisen
    },
    'export': {
        'capacity_gw': 15.0,
        'marginal_cost': -100.0,  # EUR/MWh - aktiviert nur bei negativen Preisen
    }
}

# Time Parameters
TIME_PARAMS = {
    'mvp': {
        'months': 3,
        'hours': 24 * 30 * 3,  # ca. 2160 Stunden
    },
    'v1': {
        'months': 12,
        'hours': 24 * 365,  # ca. 8760 Stunden
    }
}

# Memory Limits (GB)
MEMORY_LIMITS = {
    'total_gb': 32,
    'os_streamlit_gb': 4,
    'available_workers_gb': 28,
    'per_worker_gb': 7,
    'safe_per_worker_gb': 5,
}

# Solver Preferences
SOLVER_PREFERENCES = {
    'default': 'highs',
    'fallback_order': ['highs', 'gurobi', 'cplex', 'glpk'],
}

# File Paths (relative to project root)
PATHS = {
    'config': 'config',
    'data_raw': 'data/raw',
    'data_processed': 'data/processed',
    'results': 'data/results',
    'results_scenarios': 'data/results/scenarios',
    'results_exports': 'data/results/exports',
    'logs': 'logs',
}

# Visualization
PLOT_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
}

PLOTLY_THEME = 'plotly_white'

# Unit Conversions
UNITS = {
    'mw_to_gw': 1e-3,
    'gw_to_mw': 1e3,
    'mwh_to_gwh': 1e-3,
    'gwh_to_mwh': 1e3,
    'twh_to_gwh': 1e3,
    'gwh_to_twh': 1e-3,
}

# Scenario Types
SCENARIO_TYPES = {
    'capacity_variation': 'Kapazitäts-Variation',
    'location_analysis': 'Standort-Analyse',
    'market_strategies': 'Markt-Strategien',
    'parameter_sweep': 'Parameter-Sweep',
}

# Battery Storage Distributions
STORAGE_DISTRIBUTIONS = {
    'uniform': 'Gleichverteilt',
    'north_heavy': 'Nord-lastig',
    'south_heavy': 'Süd-lastig',
    'demand_proportional': 'Nachfrage-proportional',
}


# Helper Functions
def calculate_marginal_cost_with_co2(
    technology: str,
    co2_price_eur_per_tonne: float = None
) -> float:
    """
    Berechnet marginale Kosten inklusive CO2-Preis.

    Parameters:
    -----------
    technology : str
        Technologie (nuclear, lignite, hard_coal, ccgt, ocgt, etc.)
    co2_price_eur_per_tonne : float, optional
        CO2-Preis in EUR/t. Falls None, wird CO2_PRICE['default'] verwendet.

    Returns:
    --------
    float
        Marginale Kosten in EUR/MWh
    """
    if technology not in GENERATOR_TECHNOLOGIES:
        raise ValueError(f"Unknown technology: {technology}")

    tech_data = GENERATOR_TECHNOLOGIES[technology]

    # Basis-Kosten (Betriebskosten + Brennstoff)
    base_cost = tech_data['marginal_cost']

    # CO2-Kosten
    if co2_price_eur_per_tonne is None:
        co2_price_eur_per_tonne = CO2_PRICE['default']

    co2_factor_kg = tech_data.get('co2_factor', 0.0)
    co2_factor_tonnes = co2_factor_kg / 1000.0  # kg → t
    co2_cost = co2_factor_tonnes * co2_price_eur_per_tonne

    total_cost = base_cost + co2_cost

    return total_cost


def get_all_marginal_costs_with_co2(
    co2_price_eur_per_tonne: float = None
) -> dict:
    """
    Berechnet marginale Kosten für alle Technologien.

    Parameters:
    -----------
    co2_price_eur_per_tonne : float, optional
        CO2-Preis in EUR/t. Falls None, wird CO2_PRICE['default'] verwendet.

    Returns:
    --------
    dict
        Technology → marginale Kosten (EUR/MWh)
    """
    if co2_price_eur_per_tonne is None:
        co2_price_eur_per_tonne = CO2_PRICE['default']

    costs = {}
    for tech in GENERATOR_TECHNOLOGIES.keys():
        costs[tech] = calculate_marginal_cost_with_co2(tech, co2_price_eur_per_tonne)

    return costs
