"""
Price Validation Script

Vergleicht simulierte Strompreise mit echten Day-Ahead Preisen von SMARD.
Ziel: Validierung der Modellgüte.
"""

import sys
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging, LogContext
from src.data_processing.smard_importer import SMARDImporter
from src.network import build_copper_plate_network
import pypsa


def load_real_prices(year: int = 2025) -> pd.Series:
    """Lädt echte Day-Ahead Preise von SMARD."""
    processed_dir = Path("data/processed")
    price_file = processed_dir / "day_ahead_prices_germany_hourly.parquet"

    if not price_file.exists():
        raise FileNotFoundError(
            f"Price file not found: {price_file}\n"
            "Please run: python src/data_processing/smard_importer.py"
        )

    df = pd.read_parquet(price_file)
    return df['price_eur_per_mwh']


def run_simulation(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Führt Simulation aus und gibt Ergebnisse zurück.

    Returns:
    --------
    pd.DataFrame
        Mit Spalten: timestamp, simulated_price, demand, ...
    """
    from src.utils.logging_config import get_logger
    logger = get_logger("validation")

    logger.info(f"Running simulation: {start_date} to {end_date}")

    # Baue Netzwerk mit echten SMARD-Daten
    network = build_copper_plate_network(
        start_date=start_date,
        end_date=end_date,
        scenario_config={
            'storage': {
                'capacity_gwh': 5.0,
                'power_gw': 2.5
            }
        }
    )

    # Optimiere
    logger.info("Optimizing network...")
    status, condition = network.optimize(solver_name='cbc', pyomo=False)

    if status != "ok":
        raise RuntimeError(f"Optimization failed: {status}, {condition}")

    logger.info("Optimization successful!")

    # Extrahiere Ergebnisse
    results = pd.DataFrame({
        'timestamp': network.snapshots,
        'simulated_price': network.buses_t.marginal_price['DE'].values,
        'demand_mw': network.loads_t.p_set['demand_DE'].values
    })

    return results


def compare_prices(
    real_prices: pd.Series,
    sim_results: pd.DataFrame,
    output_dir: Path
) -> Dict:
    """
    Vergleicht simulierte mit echten Preisen.

    Returns:
    --------
    dict
        Validierungs-Metriken
    """
    from src.utils.logging_config import get_logger
    logger = get_logger("validation")

    logger.info("Comparing simulated vs. real prices...")

    # Merge
    comparison = pd.DataFrame({
        'timestamp': sim_results['timestamp'],
        'simulated': sim_results['simulated_price'],
        'real': real_prices.loc[sim_results['timestamp']].values,
        'demand': sim_results['demand_mw']
    })

    # Entferne NaN
    comparison = comparison.dropna()

    if len(comparison) == 0:
        logger.error("No overlapping data between simulation and real prices!")
        return {}

    # Berechne Metriken
    metrics = {}

    # Mean Absolute Error (MAE)
    metrics['mae'] = np.abs(comparison['simulated'] - comparison['real']).mean()

    # Mean Absolute Percentage Error (MAPE)
    # Vorsicht: MAPE ist problematisch bei Preisen nahe 0
    non_zero = comparison[comparison['real'].abs() > 1]  # Nur Preise > 1 EUR
    if len(non_zero) > 0:
        metrics['mape'] = 100 * np.abs(
            (non_zero['simulated'] - non_zero['real']) / non_zero['real']
        ).mean()
    else:
        metrics['mape'] = np.nan

    # Root Mean Squared Error (RMSE)
    metrics['rmse'] = np.sqrt(((comparison['simulated'] - comparison['real']) ** 2).mean())

    # Correlation
    metrics['correlation'] = comparison['simulated'].corr(comparison['real'])

    # R² Score
    ss_res = ((comparison['real'] - comparison['simulated']) ** 2).sum()
    ss_tot = ((comparison['real'] - comparison['real'].mean()) ** 2).sum()
    metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # Bias (systematische Über-/Unterschätzung)
    metrics['bias'] = (comparison['simulated'] - comparison['real']).mean()

    # Deskriptive Statistik
    metrics['sim_mean'] = comparison['simulated'].mean()
    metrics['sim_std'] = comparison['simulated'].std()
    metrics['sim_min'] = comparison['simulated'].min()
    metrics['sim_max'] = comparison['simulated'].max()

    metrics['real_mean'] = comparison['real'].mean()
    metrics['real_std'] = comparison['real'].std()
    metrics['real_min'] = comparison['real'].min()
    metrics['real_max'] = comparison['real'].max()

    # Log Metriken
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION METRICS")
    logger.info("=" * 80)
    logger.info(f"Data points: {len(comparison)}")
    logger.info(f"\nError Metrics:")
    logger.info(f"  MAE:         {metrics['mae']:.2f} EUR/MWh")
    logger.info(f"  RMSE:        {metrics['rmse']:.2f} EUR/MWh")
    logger.info(f"  MAPE:        {metrics['mape']:.1f}%")
    logger.info(f"  Bias:        {metrics['bias']:.2f} EUR/MWh")
    logger.info(f"\nGoodness of Fit:")
    logger.info(f"  Correlation: {metrics['correlation']:.3f}")
    logger.info(f"  R²:          {metrics['r2']:.3f}")
    logger.info(f"\nPrice Statistics:")
    logger.info(f"  Real:       {metrics['real_mean']:.1f} ± {metrics['real_std']:.1f} EUR/MWh")
    logger.info(f"  Simulated:  {metrics['sim_mean']:.1f} ± {metrics['sim_std']:.1f} EUR/MWh")

    # Speichere Vergleich als CSV
    comparison_file = output_dir / "price_comparison.csv"
    comparison.to_csv(comparison_file, index=False)
    logger.info(f"\nSaved comparison to: {comparison_file}")

    # Erstelle Visualisierungen
    create_validation_plots(comparison, metrics, output_dir)

    return metrics


def create_validation_plots(comparison: pd.DataFrame, metrics: Dict, output_dir: Path):
    """Erstellt Validierungs-Plots."""
    from src.utils.logging_config import get_logger
    logger = get_logger("validation")

    logger.info("Creating validation plots...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Price Validation: Simulated vs. Real', fontsize=16, fontweight='bold')

    # 1. Zeitreihen-Vergleich
    ax = axes[0, 0]
    ax.plot(comparison['timestamp'], comparison['real'], label='Real (SMARD)', alpha=0.7, linewidth=0.5)
    ax.plot(comparison['timestamp'], comparison['simulated'], label='Simulated (PyPSA)', alpha=0.7, linewidth=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.set_title('Price Time Series Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Scatter Plot
    ax = axes[0, 1]
    ax.scatter(comparison['real'], comparison['simulated'], alpha=0.3, s=1)
    # Perfect prediction line
    min_price = min(comparison['real'].min(), comparison['simulated'].min())
    max_price = max(comparison['real'].max(), comparison['simulated'].max())
    ax.plot([min_price, max_price], [min_price, max_price], 'r--', label='Perfect prediction')
    ax.set_xlabel('Real Price (EUR/MWh)')
    ax.set_ylabel('Simulated Price (EUR/MWh)')
    ax.set_title(f'Scatter Plot (R²={metrics["r2"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Error Distribution
    ax = axes[1, 0]
    errors = comparison['simulated'] - comparison['real']
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.axvline(errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean error ({errors.mean():.1f})')
    ax.set_xlabel('Error (Simulated - Real) [EUR/MWh]')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Price Duration Curves
    ax = axes[1, 1]
    real_sorted = comparison['real'].sort_values(ascending=False).reset_index(drop=True)
    sim_sorted = comparison['simulated'].sort_values(ascending=False).reset_index(drop=True)
    hours = np.arange(len(real_sorted))
    ax.plot(hours, real_sorted, label='Real', linewidth=2)
    ax.plot(hours, sim_sorted, label='Simulated', linewidth=2)
    ax.set_xlabel('Hours (sorted by price)')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.set_title('Price Duration Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Speichere Plot
    plot_file = output_dir / "validation_plots.png"
    plt.savefig(plot_file, dpi=150)
    logger.info(f"Saved plots to: {plot_file}")

    plt.close()


def main():
    """Hauptfunktion für Validierung."""
    logger = setup_logging(
        level="INFO",
        log_file="logs/validate_prices.log",
        console=True
    )

    logger.info("=" * 80)
    logger.info("PRICE VALIDATION: Simulated vs. Real (SMARD)")
    logger.info("=" * 80)

    # Output-Verzeichnis
    output_dir = Path("data/results/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Lade echte Preise
        with LogContext(logger, "Loading real prices from SMARD"):
            real_prices = load_real_prices(year=2025)
            logger.info(f"Loaded {len(real_prices)} hours of real prices")

        # 2. Simulation durchführen
        # Starte mit 1 Woche für schnellen Test
        start_date = "2025-01-01"
        end_date = "2025-01-07"

        with LogContext(logger, f"Running simulation ({start_date} to {end_date})"):
            sim_results = run_simulation(start_date, end_date)

        # 3. Vergleiche Preise
        with LogContext(logger, "Comparing prices"):
            metrics = compare_prices(real_prices, sim_results, output_dir)

        # 4. Interpretation
        logger.info("\n" + "=" * 80)
        logger.info("INTERPRETATION")
        logger.info("=" * 80)

        if metrics['mae'] < 10:
            logger.info("✓ EXCELLENT: MAE < 10 EUR/MWh - Modell sehr gut!")
        elif metrics['mae'] < 20:
            logger.info("✓ GOOD: MAE < 20 EUR/MWh - Modell gut!")
        elif metrics['mae'] < 50:
            logger.info("⚠ MODERATE: MAE < 50 EUR/MWh - Modell ok, Verbesserungspotenzial")
        else:
            logger.info("✗ POOR: MAE > 50 EUR/MWh - Modell braucht Verbesserung")

        if abs(metrics['bias']) < 5:
            logger.info("✓ Kein systematischer Bias")
        elif metrics['bias'] > 0:
            logger.info(f"⚠ Modell überschätzt Preise um {metrics['bias']:.1f} EUR/MWh")
        else:
            logger.info(f"⚠ Modell unterschätzt Preise um {-metrics['bias']:.1f} EUR/MWh")

        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION COMPLETE!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
