"""
Renewable Energy Profile Generator

Generiert oder lädt historische Kapazitätsfaktor-Zeitreihen für
Wind Onshore, Wind Offshore und Solar PV in Deutschland.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

from ..utils.logging_config import get_logger


logger = get_logger("data.renewable_profiles")


class RenewableProfileGenerator:
    """
    Generiert realistische Kapazitätsfaktor-Profile für Erneuerbare Energien.

    Verwendet historische Wetterdaten (wenn verfügbar) oder generiert
    synthetische Profile basierend auf typischen Mustern.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialisiert den Generator.

        Parameters:
        -----------
        data_dir : Path, optional
            Verzeichnis für Rohdaten. Wenn None, wird Projekt-Standard verwendet.
        """
        self.logger = get_logger("data.renewable_profiles")

        if data_dir is None:
            # Projekt-Wurzel finden
            current_file = Path(__file__)
            project_root = current_file.parents[2]
            data_dir = project_root / "data" / "raw"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"RenewableProfileGenerator initialized with data_dir={self.data_dir}")

    def generate_profile(
        self,
        technology: str,
        start_date: str,
        end_date: str,
        freq: str = 'h'
    ) -> pd.Series:
        """
        Generiert oder lädt Profil für eine Technologie.

        Parameters:
        -----------
        technology : str
            'wind_onshore', 'wind_offshore', oder 'solar'
        start_date : str
            Start-Datum (YYYY-MM-DD)
        end_date : str
            End-Datum (YYYY-MM-DD)
        freq : str
            Frequenz ('h' für stündlich, 'd' für täglich)

        Returns:
        --------
        pd.Series
            Kapazitätsfaktoren (0-1) mit Zeitindex
        """
        # Versuche, historische Daten zu laden
        profile = self._load_historical_profile(technology, start_date, end_date, freq)

        if profile is None:
            # Fallback: Generiere synthetisches Profil
            self.logger.warning(
                f"No historical data for {technology}, generating synthetic profile"
            )
            profile = self._generate_synthetic_profile(
                technology, start_date, end_date, freq
            )

        return profile

    def _load_historical_profile(
        self,
        technology: str,
        start_date: str,
        end_date: str,
        freq: str
    ) -> Optional[pd.Series]:
        """Versucht, historisches Profil zu laden."""
        # Suche nach vorverarbeiteten Daten
        processed_dir = self.data_dir.parent / "processed"
        profile_file = processed_dir / f"{technology}_germany_hourly.parquet"

        if not profile_file.exists():
            return None

        try:
            df = pd.read_parquet(profile_file)

            # Filter für Zeitraum
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            df = df.loc[start:end]

            # Resample falls nötig
            if freq != 'h':
                df = df.resample(freq).mean()

            self.logger.info(f"Loaded historical profile for {technology} from {profile_file}")

            return df['capacity_factor']

        except Exception as e:
            self.logger.warning(f"Failed to load historical profile: {e}")
            return None

    def _generate_synthetic_profile(
        self,
        technology: str,
        start_date: str,
        end_date: str,
        freq: str
    ) -> pd.Series:
        """
        Generiert synthetisches, aber realistisches Profil.

        Basiert auf:
        - Saisonalen Mustern (Jahreszeiten)
        - Täglichen Mustern (nur Solar)
        - Autokorrelation (Wetter-Fronten dauern mehrere Stunden)
        - Stochastischen Komponenten
        """
        # Zeitindex erstellen
        snapshots = pd.date_range(start=start_date, end=end_date, freq=freq)
        np.random.seed(42)  # Reproduzierbarkeit

        if technology == 'solar':
            profile = self._generate_solar_profile(snapshots)
        elif technology == 'wind_onshore':
            profile = self._generate_wind_onshore_profile(snapshots)
        elif technology == 'wind_offshore':
            profile = self._generate_wind_offshore_profile(snapshots)
        else:
            raise ValueError(f"Unknown technology: {technology}")

        self.logger.info(
            f"Generated synthetic {technology} profile: "
            f"{len(snapshots)} timesteps, avg CF={profile.mean():.2%}"
        )

        return pd.Series(profile, index=snapshots, name='capacity_factor')

    def _generate_solar_profile(self, snapshots: pd.DatetimeIndex) -> np.ndarray:
        """
        Generiert Solar-PV Profil.

        Eigenschaften:
        - Nur tagsüber (Sonnenaufgang bis Sonnenuntergang)
        - Maximum mittags
        - Höher im Sommer als im Winter
        - Durchschnittlicher CF: ~11% (Deutschland)
        """
        n = len(snapshots)
        profile = np.zeros(n)

        # Extrahiere Zeit-Komponenten
        hours = snapshots.hour.values
        day_of_year = snapshots.dayofyear.values

        # Saisonale Variation (Winter niedriger, Sommer höher)
        # Maximum ~21. Juni (Tag 172), Minimum ~21. Dezember (Tag 355)
        seasonal = 0.4 + 0.6 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
        seasonal = np.clip(seasonal, 0.3, 1.0)

        # Sonnenaufgang/-untergangszeiten (vereinfacht)
        # Winter: 8-16 Uhr, Sommer: 5-21 Uhr
        sunrise_hour = 8 - 3 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
        sunset_hour = 16 + 5 * np.sin((day_of_year - 80) * 2 * np.pi / 365)

        # Tägliches Muster (Glockenform)
        for i in range(n):
            if sunrise_hour[i] <= hours[i] <= sunset_hour[i]:
                # Sinus-basierte Glockenform
                day_length = sunset_hour[i] - sunrise_hour[i]
                progress = (hours[i] - sunrise_hour[i]) / day_length

                # Glockenform (Maximum bei Mittag)
                daily = np.sin(progress * np.pi) ** 2

                profile[i] = daily * seasonal[i]

        # Wolken/Wetter-Variabilität
        # Autokorrelation mit gleitendem Durchschnitt
        weather_noise = np.random.beta(5, 2, n)  # Bias zu höheren Werten
        weather_smooth = pd.Series(weather_noise).rolling(
            window=6, center=True, min_periods=1
        ).mean().values

        profile = profile * weather_smooth

        # Skaliere auf realistischen durchschnittlichen CF (~11%)
        target_cf = 0.11
        current_cf = profile.mean()

        if current_cf > 0:
            profile = profile * (target_cf / current_cf)

        profile = np.clip(profile, 0.0, 1.0)

        return profile

    def _generate_wind_onshore_profile(self, snapshots: pd.DatetimeIndex) -> np.ndarray:
        """
        Generiert Wind Onshore Profil.

        Eigenschaften:
        - Höher im Winter als im Sommer
        - Keine tägliche Periodizität
        - Stark autokorreliert (Wetterfronten)
        - Durchschnittlicher CF: ~23% (Deutschland)
        """
        n = len(snapshots)
        day_of_year = snapshots.dayofyear.values

        # Saisonale Variation (Winter höher)
        seasonal = 0.7 + 0.3 * np.cos((day_of_year - 1) * 2 * np.pi / 365)

        # Basis: Beta-Verteilung (biased zu niedrigen Werten, gelegentliche hohe Werte)
        base_wind = np.random.beta(2, 5, n)

        # Starke Autokorrelation (Wetterfronten dauern 6-24 Stunden)
        smooth_window = 12
        wind_smooth = pd.Series(base_wind).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean().values

        # Kombiniere
        profile = wind_smooth * seasonal

        # Skaliere auf realistischen CF (~23%)
        target_cf = 0.23
        current_cf = profile.mean()

        if current_cf > 0:
            profile = profile * (target_cf / current_cf)

        profile = np.clip(profile, 0.0, 1.0)

        return profile

    def _generate_wind_offshore_profile(self, snapshots: pd.DatetimeIndex) -> np.ndarray:
        """
        Generiert Wind Offshore Profil.

        Eigenschaften:
        - Ähnlich wie Onshore, aber:
          * Höhere Kapazitätsfaktoren (~40%)
          * Weniger Variabilität
          * Höhere Minimalwerte
        """
        # Starte mit Onshore-Profil
        profile = self._generate_wind_onshore_profile(snapshots)

        # Offshore ist konsistenter und stärker
        # Erhöhe Basis-Level
        profile = profile * 1.5 + 0.1

        # Glätte stärker (weniger Turbulenz auf See)
        profile = pd.Series(profile).rolling(
            window=6, center=True, min_periods=1
        ).mean().values

        # Skaliere auf realistischen CF (~40%)
        target_cf = 0.40
        current_cf = profile.mean()

        if current_cf > 0:
            profile = profile * (target_cf / current_cf)

        profile = np.clip(profile, 0.0, 1.0)

        return profile

    def generate_all_profiles(
        self,
        start_date: str,
        end_date: str,
        save_to_disk: bool = True
    ) -> Dict[str, pd.Series]:
        """
        Generiert alle Erneuerbare-Profile auf einmal.

        Parameters:
        -----------
        start_date : str
            Start-Datum (YYYY-MM-DD)
        end_date : str
            End-Datum (YYYY-MM-DD)
        save_to_disk : bool
            Ob Profile als Parquet gespeichert werden sollen

        Returns:
        --------
        dict
            Dictionary mit technology → pd.Series
        """
        technologies = ['solar', 'wind_onshore', 'wind_offshore']
        profiles = {}

        for tech in technologies:
            self.logger.info(f"Generating profile for {tech}...")
            profile = self.generate_profile(tech, start_date, end_date)
            profiles[tech] = profile

            if save_to_disk:
                self._save_profile(tech, profile)

        return profiles

    def _save_profile(self, technology: str, profile: pd.Series):
        """Speichert Profil als Parquet."""
        processed_dir = self.data_dir.parent / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        output_file = processed_dir / f"{technology}_germany_hourly.parquet"

        df = pd.DataFrame({
            'timestamp': profile.index,
            'capacity_factor': profile.values
        })
        df.set_index('timestamp', inplace=True)

        df.to_parquet(output_file)

        self.logger.info(f"Saved profile to {output_file}")

    def load_demand_profile(
        self,
        start_date: str,
        end_date: str,
        annual_demand_twh: float = 500.0,
        peak_load_gw: float = 80.0
    ) -> pd.Series:
        """
        Lädt oder generiert Nachfrage-Profil für Deutschland.

        Parameters:
        -----------
        start_date : str
            Start-Datum (YYYY-MM-DD)
        end_date : str
            End-Datum (YYYY-MM-DD)
        annual_demand_twh : float
            Jährlicher Verbrauch in TWh
        peak_load_gw : float
            Spitzenlast in GW

        Returns:
        --------
        pd.Series
            Nachfrage in MW
        """
        # Versuche, historische Daten zu laden
        processed_dir = self.data_dir.parent / "processed"
        demand_file = processed_dir / "demand_germany_hourly.parquet"

        if demand_file.exists():
            try:
                df = pd.read_parquet(demand_file)
                start = pd.Timestamp(start_date)
                end = pd.Timestamp(end_date)
                df = df.loc[start:end]

                self.logger.info(f"Loaded demand profile from {demand_file}")
                return df['demand_mw']

            except Exception as e:
                self.logger.warning(f"Failed to load demand profile: {e}")

        # Fallback: Generiere synthetisches Profil
        self.logger.warning("Generating synthetic demand profile")

        snapshots = pd.date_range(start=start_date, end=end_date, freq='h')

        # Durchschnittliche Last
        hours_per_year = 8760
        avg_load_gw = annual_demand_twh * 1000 / hours_per_year
        avg_load_mw = avg_load_gw * 1000

        demand = np.ones(len(snapshots)) * avg_load_mw

        # Tägliches Muster
        hours = snapshots.hour.values
        daily_pattern = 0.85 + 0.3 * np.sin((hours - 6) * np.pi / 12)
        daily_pattern = np.clip(daily_pattern, 0.7, 1.3)

        # Wochenmuster
        weekday = snapshots.dayofweek.values
        weekly_pattern = np.where(weekday < 5, 1.05, 0.9)

        # Saisonales Muster
        day_of_year = snapshots.dayofyear.values
        seasonal_pattern = 1.0 + 0.15 * np.cos((day_of_year - 1) * 2 * np.pi / 365)

        # Kombiniere
        demand = demand * daily_pattern * weekly_pattern * seasonal_pattern

        # Rauschen
        np.random.seed(42)
        noise = np.random.normal(1.0, 0.05, len(demand))
        demand = demand * noise

        # Clip
        demand = np.clip(demand, 0.3 * peak_load_gw * 1000, peak_load_gw * 1000)

        profile = pd.Series(demand, index=snapshots, name='demand_mw')

        # Speichere für zukünftige Verwendung
        df = pd.DataFrame({
            'timestamp': profile.index,
            'demand_mw': profile.values
        })
        df.set_index('timestamp', inplace=True)

        output_file = processed_dir / "demand_germany_hourly.parquet"
        df.to_parquet(output_file)

        self.logger.info(f"Generated and saved demand profile to {output_file}")

        return profile


def generate_profiles_for_timerange(
    start_date: str,
    end_date: str,
    data_dir: Optional[Path] = None
) -> Dict[str, pd.Series]:
    """
    Convenience-Funktion zum Generieren aller Profile.

    Parameters:
    -----------
    start_date : str
        Start-Datum (YYYY-MM-DD)
    end_date : str
        End-Datum (YYYY-MM-DD)
    data_dir : Path, optional
        Daten-Verzeichnis

    Returns:
    --------
    dict
        Dictionary mit allen Profilen
    """
    generator = RenewableProfileGenerator(data_dir=data_dir)

    profiles = generator.generate_all_profiles(
        start_date=start_date,
        end_date=end_date,
        save_to_disk=True
    )

    # Füge Nachfrage hinzu
    profiles['demand'] = generator.load_demand_profile(
        start_date=start_date,
        end_date=end_date
    )

    return profiles


# CLI für direktes Ausführen
if __name__ == "__main__":
    from ..utils.logging_config import setup_logging
    import sys

    setup_logging(level="INFO", console=True)

    # Parse Command-Line-Arguments
    if len(sys.argv) >= 3:
        start = sys.argv[1]
        end = sys.argv[2]
    else:
        # Default: 3 Monate für MVP
        start = "2024-01-01"
        end = "2024-03-31"

    logger.info(f"Generating profiles for {start} to {end}")

    profiles = generate_profiles_for_timerange(start, end)

    print("\n" + "=" * 60)
    print("Generated Profiles Summary:")
    print("=" * 60)

    for tech, profile in profiles.items():
        if tech == 'demand':
            print(f"\n{tech.upper()}:")
            print(f"  Length: {len(profile)} hours")
            print(f"  Average: {profile.mean():.1f} MW")
            print(f"  Peak: {profile.max():.1f} MW")
            print(f"  Min: {profile.min():.1f} MW")
        else:
            print(f"\n{tech.upper()}:")
            print(f"  Length: {len(profile)} hours")
            print(f"  Average CF: {profile.mean():.2%}")
            print(f"  Max CF: {profile.max():.2%}")
            print(f"  Min CF: {profile.min():.2%}")
