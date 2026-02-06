# German Power Market Simulation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

Ein PyPSA-basiertes Simulationsmodell für den deutschen Strommarkt mit Fokus auf Batteriespeicher-Kannibalisierung.

## Features

- 🔋 **Batteriespeicher-Kannibalisierung**: Analyse wie zusätzliche Speicherkapazität Erlöse reduziert
- ⚡ **Merit Order Dispatch**: Copper Plate Modell für Deutschland
- 📊 **SMARD-Daten Integration**: Echte Erzeugungsprofile (Wind, Solar, Nachfrage)
- 💰 **CO2-Pricing**: EU ETS Preise integriert (~75 EUR/t)
- 📈 **Validierung**: Korrelation 0.867 mit echten SMARD-Preisen

## Modell-Validierung

- **Korrelation**: 0.867 mit SMARD Day-Ahead Preisen ✅
- **MAE**: ~26 EUR/MWh (systematischer Bias)
- **Geeignet für**: Relative Kannibalisierungs-Analysen, Szenario-Vergleiche
- **Nicht geeignet für**: Absolute Erlös-Prognosen (ohne Korrekturfaktor)

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/power-market-sim.git
cd power-market-sim
pip install -r requirements.txt
```

## Usage

### Dashboard starten

```bash
streamlit run dashboard/app.py
```

### Kannibalisierungs-Analyse

1. Zeitraum wählen (empfohlen: ganzes Jahr)
2. Erzeugungspark konfigurieren
3. Tab "Kannibalisierung" öffnen
4. Speicher-Schritte konfigurieren (z.B. 5 GWh, max 50 GWh)
5. Analyse starten
6. Ergebnisse downloaden (CSV)

## Modell-Architektur

```
Copper Plate Modell (Deutschland)
├── Merit Order Dispatch (Linear Programming)
├── CO2-Pricing (75 EUR/t)
├── Speicher-Effizienz (90% Round-trip)
├── Ramping-Constraints (5-30% pro Stunde)
└── Cross-Border Flows (gewichtete Import/Export-Preise)
```

## Limitationen

⚠️ **Systematischer Bias**: Preise ~39% zu niedrig (Copper Plate, kein Scarcity Pricing)  
⚠️ **Perfekte Voraussicht**: Überschätzt Speicher-Erlöse um 20-40%  
✅ **Relative Effekte valide**: Kannibalisierungs-Trends sind robust

## Forschungs-Ergebnisse

**Storage Cannibalization Effect (2025):**
- 5 GWh: 37.1 k€/MWh/Jahr
- 50 GWh: 16.2 k€/MWh/Jahr (-56%)
- 100 GWh: 10.2 k€/MWh/Jahr (-73%)

→ **Massive Kannibalisierung** ab ~25-50 GWh Gesamtkapazität

## Datenquellen

- [SMARD](https://www.smard.de/) - Bundesnetzagentur
- RWTH Aachen / ISEA (Validierung)
- EU ETS CO2-Preise

## Technologie-Stack

- **PyPSA**: Power System Analysis
- **Streamlit**: Interactive Dashboard
- **Plotly**: Visualisierungen
- **Pandas**: Datenverarbeitung
- **HiGHS**: LP-Solver

## Lizenz

MIT License

## Kontakt

Für Fragen zur Forschung oder Modell-Details: [Your Contact]
