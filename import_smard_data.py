"""
SMARD Data Import Script

Importiert SMARD-Daten aus CSV-Dateien.
"""

import sys
from pathlib import Path

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging
from src.data_processing.smard_importer_v2 import SMARDImporterV2


def main():
    """Hauptfunktion für SMARD-Import."""

    # Setup Logging
    logger = setup_logging(
        level="INFO",
        log_file="logs/smard_import.log",
        console=True
    )

    logger.info("=" * 80)
    logger.info("SMARD DATA IMPORT")
    logger.info("=" * 80)

    # Importiere Daten
    importer = SMARDImporterV2()
    data = importer.import_all()

    if len(data) > 0:
        print("\n" + "=" * 80)
        print("✓ IMPORT SUCCESS!")
        print("=" * 80)
        print(f"\n Imported {len(data)} datasets:")

        for name in sorted(data.keys()):
            df = data[name]
            print(f"  ✓ {name:20s} - {len(df):6d} hours")

        print(f"\n Processed files saved to:")
        print(f"  {importer.processed_dir}")

        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("1. Run validation: python validate_prices.py")
        print("2. Or run simulation: python test_network_build.py")

    else:
        print("\n" + "=" * 80)
        print("✗ ERROR: No data imported!")
        print("=" * 80)
        print("\nPlease check:")
        print(f"1. CSV files exist in: {importer.raw_data_dir}")
        print("2. Files have SMARD format (Semikolon-getrennt)")
        print("3. Check logs/smard_import.log for details")

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
