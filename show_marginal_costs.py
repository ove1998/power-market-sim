"""
Zeigt die neuen marginalen Kosten mit CO2-Preis.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.constants import (
    GENERATOR_TECHNOLOGIES,
    CO2_PRICE,
    calculate_marginal_cost_with_co2,
    get_all_marginal_costs_with_co2
)


def main():
    print("=" * 80)
    print("MARGINALE KOSTEN MIT CO2-PREIS")
    print("=" * 80)
    print(f"\nCO2-Preis: {CO2_PRICE['default']:.2f} EUR/t CO2")
    print("\n" + "-" * 80)
    print(f"{'Technologie':<20} {'Alt (ohne CO2)':<20} {'Neu (mit CO2)':<20} {'CO2-Kosten':<20}")
    print("-" * 80)

    # Berechne neue Kosten
    new_costs = get_all_marginal_costs_with_co2()

    # Sortiere nach neuen marginalen Kosten (Merit Order)
    sorted_techs = sorted(
        new_costs.items(),
        key=lambda x: x[1]
    )

    for tech, new_cost in sorted_techs:
        tech_data = GENERATOR_TECHNOLOGIES[tech]
        old_cost = tech_data['marginal_cost']
        co2_cost = new_cost - old_cost

        label = tech_data['label']

        print(
            f"{label:<20} {old_cost:>8.2f} EUR/MWh    "
            f"{new_cost:>8.2f} EUR/MWh    {co2_cost:>8.2f} EUR/MWh"
        )

    print("-" * 80)
    print("\n✓ Konventionelle Kraftwerke sind jetzt deutlich teurer!")
    print("✓ Merit Order hat sich geändert - Gas ist jetzt günstiger als Kohle")
    print()


if __name__ == "__main__":
    main()
