"""
Test Script: Build and Optimize First Copper Plate Network

Testet den vollständigen Workflow:
1. Profile generieren
2. Netzwerk bauen
3. Optimierung durchführen
4. Ergebnisse anzeigen
"""

import sys
from pathlib import Path

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging, LogContext
from src.data_processing.renewable_profiles import generate_profiles_for_timerange
from src.network import build_copper_plate_network, get_generation_mix_summary
import pandas as pd


def main():
    """Hauptfunktion für den Test."""

    # Setup Logging
    logger = setup_logging(
        level="INFO",
        log_file="logs/test_network_build.log",
        console=True
    )

    logger.info("=" * 80)
    logger.info("TEST: Build and Optimize First Copper Plate Network")
    logger.info("=" * 80)

    # Phase 1: Profile generieren
    with LogContext(logger, "Generating renewable & demand profiles", log_memory=True):
        profiles = generate_profiles_for_timerange(
            start_date="2024-01-01",
            end_date="2024-03-31"  # 3 Monate (MVP)
        )

        logger.info(f"Generated {len(profiles)} profiles:")
        for tech, profile in profiles.items():
            if tech == 'demand':
                logger.info(
                    f"  - {tech}: {len(profile)} hours, "
                    f"avg={profile.mean():.1f} MW, peak={profile.max():.1f} MW"
                )
            else:
                logger.info(
                    f"  - {tech}: {len(profile)} hours, avg CF={profile.mean():.2%}"
                )

    # Phase 2: Netzwerk bauen
    with LogContext(logger, "Building Copper Plate Network", log_memory=True):
        # Szenario mit Batteriespeicher
        scenario_config = {
            'storage': {
                'capacity_gwh': 5.0,  # 5 GWh Kapazität
                'power_gw': 2.5       # 2.5 GW Leistung (C-Rate = 0.5)
            }
        }

        network = build_copper_plate_network(
            start_date="2024-01-01",
            end_date="2024-03-31",  # 3 Monate (MVP)
            scenario_config=scenario_config
        )

        logger.info(f"Network built successfully!")

    # Phase 3: Network-Zusammenfassung
    logger.info("\n" + "-" * 80)
    logger.info("NETWORK SUMMARY")
    logger.info("-" * 80)

    # Busse
    logger.info(f"\nBuses: {len(network.buses)}")
    for bus_name in network.buses.index:
        logger.info(f"  - {bus_name}")

    # Generatoren
    logger.info(f"\nGenerators: {len(network.generators)}")
    gen_summary = get_generation_mix_summary(network)
    logger.info("\nGeneration Mix:")
    logger.info(gen_summary.to_string(index=False))

    # Speicher
    if len(network.storage_units) > 0:
        logger.info(f"\nStorage Units: {len(network.storage_units)}")
        for storage_name, storage in network.storage_units.iterrows():
            logger.info(
                f"  - {storage_name}: "
                f"{storage['p_nom']:.0f} MW, "
                f"{storage['max_hours']:.1f} hours "
                f"({storage['p_nom'] * storage['max_hours'] / 1000:.1f} GWh)"
            )

    # Last
    logger.info(f"\nLoads: {len(network.loads)}")
    for load_name, load in network.loads.iterrows():
        avg_demand = network.loads_t.p_set[load_name].mean()
        peak_demand = network.loads_t.p_set[load_name].max()
        logger.info(
            f"  - {load_name}: avg={avg_demand:.0f} MW, peak={peak_demand:.0f} MW"
        )

    # Phase 4: Optimierung durchführen
    with LogContext(logger, "Optimizing Network (PyPSA Linear Optimal Power Flow)", log_memory=True):
        try:
            # Optimiere mit CBC Solver (Linopy Backend)
            status, condition = network.optimize(
                solver_name='cbc',
                pyomo=False  # Linopy Backend (70% weniger Speicher!)
            )

            if status == "ok":
                logger.info("✓ Optimization successful!")
            else:
                logger.error(f"✗ Optimization failed: {status}, {condition}")
                return

        except Exception as e:
            logger.error(f"✗ Optimization error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return

    # Phase 5: Ergebnisse analysieren
    logger.info("\n" + "-" * 80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("-" * 80)

    # Objektive (Total System Cost)
    objective = network.objective
    logger.info(f"\nTotal System Cost: {objective:,.2f} EUR")

    # Strompreise (Marginal Price am Bus "DE")
    if hasattr(network.buses_t, 'marginal_price') and 'DE' in network.buses_t.marginal_price.columns:
        prices = network.buses_t.marginal_price['DE']

        logger.info(f"\nElectricity Prices (Bus DE):")
        logger.info(f"  - Mean: {prices.mean():.2f} EUR/MWh")
        logger.info(f"  - Median: {prices.median():.2f} EUR/MWh")
        logger.info(f"  - Min: {prices.min():.2f} EUR/MWh")
        logger.info(f"  - Max: {prices.max():.2f} EUR/MWh")
        logger.info(f"  - Std Dev: {prices.std():.2f} EUR/MWh")

        # Preisdauerkurve
        price_duration = prices.sort_values(ascending=False).reset_index(drop=True)

        logger.info(f"\nPrice Duration Curve (Quantiles):")
        for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            value = prices.quantile(q)
            logger.info(f"  - {q:.0%}: {value:.2f} EUR/MWh")

    # Generator Dispatch
    if hasattr(network.generators_t, 'p'):
        logger.info(f"\nGenerator Dispatch (Total Energy):")

        for gen_name in network.generators.index:
            if gen_name in network.generators_t.p.columns:
                dispatch_mwh = network.generators_t.p[gen_name].sum()
                dispatch_gwh = dispatch_mwh / 1000

                carrier = network.generators.at[gen_name, 'carrier']
                capacity_mw = network.generators.at[gen_name, 'p_nom']

                # Capacity Factor
                max_possible_mwh = capacity_mw * len(network.snapshots)
                cf = dispatch_mwh / max_possible_mwh if max_possible_mwh > 0 else 0

                logger.info(
                    f"  - {gen_name}: {dispatch_gwh:.2f} GWh (CF={cf:.1%})"
                )

    # Speicher-Verhalten
    if len(network.storage_units) > 0 and hasattr(network.storage_units_t, 'p'):
        logger.info(f"\nStorage Behavior:")

        for storage_name in network.storage_units.index:
            if storage_name in network.storage_units_t.p.columns:
                p_series = network.storage_units_t.p[storage_name]
                soc_series = network.storage_units_t.state_of_charge[storage_name]

                # Charging/Discharging
                charging = p_series[p_series < 0].abs().sum()  # MWh
                discharging = p_series[p_series > 0].sum()  # MWh

                # State of Charge
                avg_soc = soc_series.mean()
                max_soc = soc_series.max()

                logger.info(f"  - {storage_name}:")
                logger.info(f"      Charged: {charging / 1000:.2f} GWh")
                logger.info(f"      Discharged: {discharging / 1000:.2f} GWh")
                logger.info(f"      Avg SoC: {avg_soc / 1000:.2f} GWh")
                logger.info(f"      Max SoC: {max_soc / 1000:.2f} GWh")

                # Round-trip efficiency check
                if charging > 0:
                    efficiency = discharging / charging
                    logger.info(f"      Effective efficiency: {efficiency:.1%}")

    # Phase 6: Export Ergebnisse (optional)
    output_dir = Path("data/results/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Speichere Preise
    if hasattr(network.buses_t, 'marginal_price') and 'DE' in network.buses_t.marginal_price.columns:
        prices_df = pd.DataFrame({
            'timestamp': network.snapshots,
            'price_eur_per_mwh': network.buses_t.marginal_price['DE'].values
        })
        prices_file = output_dir / "electricity_prices.csv"
        prices_df.to_csv(prices_file, index=False)
        logger.info(f"\n✓ Exported prices to {prices_file}")

    # Speichere Generator Dispatch
    if hasattr(network.generators_t, 'p'):
        dispatch_df = network.generators_t.p.copy()
        dispatch_df.index.name = 'timestamp'
        dispatch_file = output_dir / "generator_dispatch.csv"
        dispatch_df.to_csv(dispatch_file)
        logger.info(f"✓ Exported generator dispatch to {dispatch_file}")

    # Speichere Storage Behavior
    if len(network.storage_units) > 0 and hasattr(network.storage_units_t, 'p'):
        storage_df = pd.DataFrame({
            'timestamp': network.snapshots,
            'p_mw': network.storage_units_t.p[network.storage_units.index[0]].values,
            'soc_mwh': network.storage_units_t.state_of_charge[network.storage_units.index[0]].values
        })
        storage_file = output_dir / "storage_behavior.csv"
        storage_df.to_csv(storage_file, index=False)
        logger.info(f"✓ Exported storage behavior to {storage_file}")

    # Finale
    logger.info("\n" + "=" * 80)
    logger.info("✓ TEST COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("  1. Extend to 3 months simulation")
    logger.info("  2. Implement parallel scenario execution")
    logger.info("  3. Build Streamlit dashboard")
    logger.info("  4. Add cannibalization analysis")


if __name__ == "__main__":
    main()
