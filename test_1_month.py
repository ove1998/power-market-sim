"""
1-Monats-Test mit echten SMARD-Daten und korrekten Kapazitäten

Testet das Modell mit Januar 2025 und vergleicht mit echten Preisen.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging
from src.network import build_copper_plate_network


def main():
    """Hauptfunktion für 1-Monats-Test."""
    logger = setup_logging(
        level="INFO",
        log_file="logs/test_1_month.log",
        console=True
    )

    logger.info("="*80)
    logger.info("1-MONTH TEST: JANUARY 2025")
    logger.info("="*80)

    # Baue Netzwerk mit echten SMARD-Daten für Januar 2025
    print("\n" + "="*80)
    print("BUILDING NETWORK WITH REAL DATA")
    print("="*80)
    print("Period: 2025-01-01 to 2025-01-31 (1 month)")

    network = build_copper_plate_network(
        start_date="2025-01-01",
        end_date="2025-01-31",
        config_path="config/default_config.yaml",
        scenario_config={
            'storage': {
                'capacity_gwh': 10.0,
                'power_gw': 5.0
            }
        }
    )

    # Zeige Netzwerk-Info
    print("\n" + "="*80)
    print("NETWORK OVERVIEW")
    print("="*80)
    print(network)

    print("\nGenerators (Top 10 by capacity):")
    top_gens = network.generators.nlargest(10, 'p_nom')
    for idx, gen in top_gens.iterrows():
        print(f"  {idx:20s}: {gen['p_nom']/1000:6.1f} GW @ {gen['marginal_cost']:6.1f} EUR/MWh")

    print("\nDemand:")
    for idx, load in network.loads.iterrows():
        avg_load = network.loads_t.p_set[idx].mean()
        peak_load = network.loads_t.p_set[idx].max()
        print(f"  {idx:20s}: avg={avg_load/1000:6.1f} GW, peak={peak_load/1000:6.1f} GW")

    # Optimiere
    print("\n" + "="*80)
    print("OPTIMIZING NETWORK")
    print("="*80)
    print("Solver: CBC")
    print("Snapshots: 745 hours (31 days)")

    logger.info("Starting optimization...")

    try:
        status, condition = network.optimize(
            solver_name='cbc',
            pyomo=False
        )

        if status != "ok":
            logger.error(f"Optimization failed: {status}, {condition}")
            return 1

        logger.info("Optimization successful!")

    except Exception as e:
        logger.error(f"Optimization error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    # Analysiere Ergebnisse
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)

    # System-Kosten
    total_cost = network.objective / 1e6  # Mio EUR
    print(f"\nTotal System Cost: {total_cost:.1f} Mio EUR")

    # Strompreise
    prices = network.buses_t.marginal_price['DE']
    print(f"\nElectricity Prices:")
    print(f"  Average:  {prices.mean():8.2f} EUR/MWh")
    print(f"  Median:   {prices.median():8.2f} EUR/MWh")
    print(f"  Std Dev:  {prices.std():8.2f} EUR/MWh")
    print(f"  Min:      {prices.min():8.2f} EUR/MWh")
    print(f"  Max:      {prices.max():8.2f} EUR/MWh")

    # Lade echte Preise zum Vergleich
    try:
        price_file = Path("data/processed/day_ahead_prices_germany_hourly.parquet")
        if price_file.exists():
            real_prices_df = pd.read_parquet(price_file)
            real_prices = real_prices_df.loc["2025-01-01":"2025-01-31", 'price_eur_per_mwh']

            print(f"\nReal SMARD Prices (same period):")
            print(f"  Average:  {real_prices.mean():8.2f} EUR/MWh")
            print(f"  Median:   {real_prices.median():8.2f} EUR/MWh")
            print(f"  Std Dev:  {real_prices.std():8.2f} EUR/MWh")
            print(f"  Min:      {real_prices.min():8.2f} EUR/MWh")
            print(f"  Max:      {real_prices.max():8.2f} EUR/MWh")

            # Vergleich
            mae = np.abs(prices - real_prices.values).mean()
            rmse = np.sqrt(((prices - real_prices.values) ** 2).mean())
            correlation = np.corrcoef(prices, real_prices.values)[0, 1]

            print(f"\nValidation Metrics:")
            print(f"  MAE (Mean Absolute Error):  {mae:.2f} EUR/MWh")
            print(f"  RMSE:                       {rmse:.2f} EUR/MWh")
            print(f"  Correlation:                {correlation:.3f}")

    except Exception as e:
        logger.warning(f"Could not load real prices for comparison: {e}")

    # Erzeugung nach Technologie
    print(f"\nGeneration by Technology (avg GW):")
    for gen in network.generators.index:
        if gen in network.generators_t.p.columns:
            avg_gen = network.generators_t.p[gen].mean() / 1000  # GW
            capacity = network.generators.loc[gen, 'p_nom'] / 1000  # GW
            cf = avg_gen / capacity if capacity > 0 else 0
            if avg_gen > 0.1:  # Nur Technologien mit > 0.1 GW avg zeigen
                print(f"  {gen:20s}: {avg_gen:6.1f} GW (CF={cf:6.1%})")

    # Speicher-Analyse
    if len(network.storage_units) > 0:
        print(f"\nBattery Storage Analysis:")
        battery_dispatch = network.storage_units_t.p['battery_DE']
        battery_soc = network.storage_units_t.state_of_charge['battery_DE']

        charging = battery_dispatch[battery_dispatch < 0].sum() * -1  # MWh
        discharging = battery_dispatch[battery_dispatch > 0].sum()  # MWh

        print(f"  Total Charged:     {charging/1000:10.1f} GWh")
        print(f"  Total Discharged:  {discharging/1000:10.1f} GWh")
        print(f"  Avg SoC:           {battery_soc.mean()/1000:10.1f} GWh")
        print(f"  Max SoC:           {battery_soc.max()/1000:10.1f} GWh")

        # Zyklen
        capacity_gwh = network.storage_units.loc['battery_DE', 'p_nom'] * \
                      network.storage_units.loc['battery_DE', 'max_hours'] / 1000
        cycles = discharging / (capacity_gwh * 1000)
        print(f"  Full Cycles:       {cycles:10.1f}")

    print("\n" + "="*80)
    print("TEST COMPLETED!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run for 3 months if results look good")
    print("2. Run validate_prices.py for detailed comparison")
    print("3. Analyze why prices differ from reality")

    return 0


if __name__ == "__main__":
    sys.exit(main())
