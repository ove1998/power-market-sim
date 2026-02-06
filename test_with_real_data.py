"""
Test Network Build mit echten SMARD-Daten

Testet das Copper Plate Modell mit echten 2025 SMARD-Daten.
Startet mit 1 Woche für schnellen Test.
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


def check_available_data():
    """Prüft, welche Daten verfügbar sind."""
    data_dir = Path("data/processed")

    print("\n" + "="*80)
    print("AVAILABLE SMARD DATA")
    print("="*80)

    files = list(data_dir.glob("*.parquet"))

    for file in sorted(files):
        df = pd.read_parquet(file)
        print(f"\n{file.name}:")
        print(f"  Rows: {len(df)}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Columns: {', '.join(df.columns)}")

        # Zeige Statistiken
        if 'capacity_factor' in df.columns:
            cf = df['capacity_factor'].dropna()
            if len(cf) > 0:
                print(f"  Capacity Factor: avg={cf.mean():.2%}, max={cf.max():.2%}")

        if 'generation_mw' in df.columns:
            gen = df['generation_mw'].dropna()
            if len(gen) > 0:
                print(f"  Generation: avg={gen.mean():.0f} MW, max={gen.max():.0f} MW")

        if 'demand_mw' in df.columns:
            demand = df['demand_mw'].dropna()
            if len(demand) > 0:
                print(f"  Demand: avg={demand.mean():.0f} MW, peak={demand.max():.0f} MW")

        if 'price_eur_per_mwh' in df.columns:
            prices = df['price_eur_per_mwh'].dropna()
            if len(prices) > 0:
                print(f"  Price: avg={prices.mean():.2f} EUR/MWh, range=[{prices.min():.2f}, {prices.max():.2f}]")


def main():
    """Hauptfunktion für Test mit echten Daten."""
    logger = setup_logging(
        level="INFO",
        log_file="logs/test_real_data.log",
        console=True
    )

    logger.info("="*80)
    logger.info("TEST WITH REAL SMARD DATA")
    logger.info("="*80)

    # Prüfe verfügbare Daten
    check_available_data()

    # Baue Netzwerk mit echten SMARD-Daten
    # Start mit 1 Woche (2025-01-01 bis 2025-01-07)
    print("\n" + "="*80)
    print("BUILDING NETWORK WITH REAL DATA")
    print("="*80)
    print("Period: 2025-01-01 to 2025-01-07 (1 week)")

    network = build_copper_plate_network(
        start_date="2025-01-01",
        end_date="2025-01-07",
        scenario_config={
            'storage': {
                'capacity_gwh': 5.0,
                'power_gw': 2.5
            }
        }
    )

    # Zeige Netzwerk-Info
    print("\n" + "="*80)
    print("NETWORK OVERVIEW")
    print("="*80)
    print(network)

    print("\nGenerators:")
    for idx, gen in network.generators.iterrows():
        print(f"  {idx:20s}: {gen['p_nom']:8.0f} MW @ {gen['marginal_cost']:6.1f} EUR/MWh")

    print("\nLoads:")
    for idx, load in network.loads.iterrows():
        avg_load = network.loads_t.p_set[idx].mean()
        peak_load = network.loads_t.p_set[idx].max()
        print(f"  {idx:20s}: avg={avg_load:8.0f} MW, peak={peak_load:8.0f} MW")

    if len(network.storage_units) > 0:
        print("\nStorage Units:")
        for idx, storage in network.storage_units.iterrows():
            print(f"  {idx:20s}: {storage['p_nom']:8.0f} MW, {storage['max_hours']:.1f} h capacity")

    # Optimiere
    print("\n" + "="*80)
    print("OPTIMIZING NETWORK")
    print("="*80)
    print("Solver: CBC")
    print("Backend: Linopy (pyomo=False)")

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
            real_prices = real_prices_df.loc["2025-01-01":"2025-01-07", 'price_eur_per_mwh']

            print(f"\nReal SMARD Prices (same period):")
            print(f"  Average:  {real_prices.mean():8.2f} EUR/MWh")
            print(f"  Median:   {real_prices.median():8.2f} EUR/MWh")
            print(f"  Std Dev:  {real_prices.std():8.2f} EUR/MWh")
            print(f"  Min:      {real_prices.min():8.2f} EUR/MWh")
            print(f"  Max:      {real_prices.max():8.2f} EUR/MWh")

            # Vergleich
            mae = np.abs(prices - real_prices).mean()
            print(f"\nMean Absolute Error: {mae:.2f} EUR/MWh")

    except Exception as e:
        logger.warning(f"Could not load real prices for comparison: {e}")

    # Erzeugung nach Technologie
    print(f"\nGeneration by Technology (avg MW):")
    for gen in network.generators.index:
        if gen in network.generators_t.p.columns:
            avg_gen = network.generators_t.p[gen].mean()
            capacity = network.generators.loc[gen, 'p_nom']
            cf = avg_gen / capacity if capacity > 0 else 0
            print(f"  {gen:20s}: {avg_gen:8.0f} MW (CF={cf:6.1%})")

    # Speicher-Analyse (falls vorhanden)
    if len(network.storage_units) > 0:
        print(f"\nBattery Storage Analysis:")
        battery_dispatch = network.storage_units_t.p['battery_DE']
        battery_soc = network.storage_units_t.state_of_charge['battery_DE']

        charging = battery_dispatch[battery_dispatch < 0].sum() * -1  # MWh
        discharging = battery_dispatch[battery_dispatch > 0].sum()  # MWh

        print(f"  Total Charged:     {charging:10.0f} MWh")
        print(f"  Total Discharged:  {discharging:10.0f} MWh")
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
    print("1. Check if simulated prices are realistic")
    print("2. Run for longer periods (1 month, 3 months)")
    print("3. Run validate_prices.py for detailed comparison")

    return 0


if __name__ == "__main__":
    sys.exit(main())
