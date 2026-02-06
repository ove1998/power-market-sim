# Strommarkt-Simulation Dashboard

## 📋 Übersicht

Interaktives Streamlit-Dashboard zur Konfiguration und Ausführung von PyPSA-Stromsystem-Simulationen für Deutschland.

**Hauptfeatures:**
- ⚙️ **Flexible Eingabe:** Erzeugungskapazitäten, Batteriespeicher, Nachfrage-Skalierung
- ⚡ **Echte SMARD-Daten:** Wind/Solar-Profile und Day-Ahead-Preise aus 2025
- 📊 **Interaktive Visualisierung:** Strompreise, Erzeugungsmix, Speicherverhalten
- ✅ **Validierung:** Vergleich simulierter mit echten SMARD-Preisen

## 🚀 Dashboard starten

```bash
# Navigiere ins Projektverzeichnis
cd C:\Users\ovekn\KES\14_Sonstiges\Ideen\claude\power-market-sim

# Starte das Dashboard
streamlit run dashboard/app.py
```

Das Dashboard öffnet sich automatisch im Browser unter `http://localhost:8501`

## 🎯 Verwendung

### 1. **Zeitraum wählen**
- Start- und Enddatum für die Simulation
- Empfohlen für erste Tests: **1 Woche - 1 Monat**
- Kürzere Zeiträume = schnellere Simulationen

### 2. **Erzeugungspark konfigurieren**
Die Sidebar zeigt Slider für alle Technologien:

**Konventionelle:**
- Kernkraft (0-10 GW) - Marginal Cost: 5 EUR/MWh
- Braunkohle (0-30 GW) - MC: 95 EUR/MWh (inkl. CO2)
- Steinkohle (0-30 GW) - MC: 107 EUR/MWh (inkl. CO2)
- Gas GuD/CCGT (0-50 GW) - MC: 92 EUR/MWh (inkl. CO2)
- Gas GT/OCGT (0-10 GW) - MC: 140 EUR/MWh (inkl. CO2)

**Erneuerbare:**
- Wind Onshore (0-120 GW) - MC: 0 EUR/MWh
- Wind Offshore (0-30 GW) - MC: 0 EUR/MWh
- Solar (0-150 GW) - MC: 0 EUR/MWh
- Laufwasser (0-10 GW) - MC: 0 EUR/MWh
- Pumpspeicher (0-20 GW) - MC: 0 EUR/MWh

> **Hinweis:** Die Grenzkosten sind im Modell fest konfiguriert und beinhalten CO2-Preise (~80 EUR/t)

### 3. **Batteriespeicher konfigurieren**
- **Kapazität (GWh):** Energiekapazität des Speichers (0-100 GWh)
- **Leistung (GW):** Maximale Lade-/Entladeleistung (0-50 GW)
- **E/P-Verhältnis:** Wird automatisch berechnet (Stunden Volllast)

### 4. **Nachfrage skalieren**
- Skalierungsfaktor: **50%-150%**
- 100% = Historische SMARD-Nachfrage
- 80% = 20% weniger Nachfrage
- 120% = 20% mehr Nachfrage

### 5. **Simulation starten**
- Button **"🚀 Simulation starten"** in der Sidebar klicken
- Fortschrittsanzeige beobachten
- Ergebnisse erscheinen automatisch nach Abschluss

## 📊 Ergebnis-Tabs

### Tab 1: 📈 Strompreise
- **Statistiken:** Durchschnitt, Median, Min, Max, Std.Abw.
- **Zeitverlauf:** Stundenweise Strompreise
- **Preisdauerkurve:** Sortierte Preise über alle Stunden
- **Histogramm:** Preisverteilung

### Tab 2: 🏭 Erzeugung
- **Gestapelte Zeitreihe:** Erzeugungsmix nach Technologie
- **Statistik-Tabelle:** Durchschnitt, Kapazität, Kapazitätsfaktoren
- **Kapazitätsfaktoren:** Bar-Chart nach Technologie

### Tab 3: 🔋 Batteriespeicher
- **Statistiken:** Geladen/Entladen (GWh), Vollzyklen, Effizienz
- **Dispatch:** Lade-/Entladevorgänge über Zeit
- **State of Charge (SoC):** Ladezustand über Zeit
- **Dispatch vs. Preis:** Vergleich Speicherverhalten mit Strompreisen

### Tab 4: 📊 Validierung
- **Vergleich:** Simulierte vs. echte SMARD-Preise
- **Fehlermetriken:** MAE, RMSE, Korrelation
- **Zeitreihen-Vergleich:** Überlagerte Plots
- **Scatter-Plot:** Korrelationsanalyse

## 🧠 Modell-Details

### Copper Plate Modell
- **Keine geografischen Netzwerk-Constraints**
- **Unbegrenzte Übertragungskapazität** zwischen allen Punkten
- **Merit Order Dispatch:** Kraftwerke nach Grenzkosten sortiert
- Fokus auf **Preis-Kannibalisierung** durch Batteriespeicher

### Merit Order (Grenzkosten aufsteigend)
1. Erneuerbare (0 EUR/MWh): Wind, Solar, Hydro
2. Kernkraft (5 EUR/MWh)
3. Braunkohle (95 EUR/MWh inkl. CO2)
4. Gas GuD (92 EUR/MWh inkl. CO2)
5. Steinkohle (107 EUR/MWh inkl. CO2)
6. Gas GT (140 EUR/MWh inkl. CO2)

**CO2-Preis:** ~80 EUR/t (2025 ETS-Preis)

### Datenquellen
- **Wind/Solar-Profile:** SMARD 2025 (Bundesnetzagentur)
- **Nachfrage:** SMARD 2025 Realisierter Stromverbrauch
- **Validierung:** SMARD 2025 Day-Ahead Preise

### Solver
- **CBC:** Open-Source LP-Solver
- **Backend:** Linopy (nicht Pyomo) für 70% weniger RAM

## 📈 Typische Anwendungsfälle

### 1. **Baseline-Simulation (Status Quo 2025)**
- Standard-Kapazitäten aus SMARD
- Batteriespeicher: 10 GWh / 5 GW
- Nachfrage: 100%
- **Ziel:** Vergleich mit echten 2025 Preisen

### 2. **Battery Storage Sweep**
- Variiere Batteriekapazität: 0, 10, 20, 50, 100 GWh
- Beobachte Preis-Kannibalisierung
- **Frage:** Ab wann fallen Preise signifikant?

### 3. **Renewable Expansion**
- Erhöhe Wind Onshore: 63 → 100 GW
- Erhöhe Solar: 87 → 150 GW
- **Frage:** Wie ändern sich Preise und Speicherbedarf?

### 4. **Demand Shock**
- Skaliere Nachfrage: 120% (z.B. Elektromobilität)
- **Frage:** Reicht die Erzeugungskapazität?

## ⚠️ Bekannte Einschränkungen

1. **Copper Plate:** Keine Netzengpässe, keine regionalen Preisunterschiede
2. **Keine Flexibilität:** Kraftwerke können ohne Verzögerung hoch/runterfahren
3. **Vereinfachte Importe:** Feste Grenzkosten statt echter europäischer Kopplung
4. **RAM-Limitiert:** Lange Zeiträume (>3 Monate) können langsam sein

## 🔧 Technische Anforderungen

**Software:**
- Python >= 3.10
- PyPSA >= 0.27.0
- Streamlit >= 1.30.0
- CBC Solver (installiert)

**Hardware:**
- RAM: Mindestens 8 GB (16 GB empfohlen)
- CPU: Multi-Core für schnellere Optimierung
- Speicher: ~2 GB für Daten und Ergebnisse

## 📝 Tipps & Tricks

### Performance
- **Kurze Zeiträume:** Starte mit 1 Woche für schnelle Tests
- **Weniger Snapshots:** Stündliche Auflösung notwendig, aber 1 Monat optimal
- **RAM-Monitoring:** Task-Manager beobachten bei langen Simulationen

### Realistische Ergebnisse
- **CO2-Preise:** Bereits in Grenzkosten enthalten (siehe Merit Order)
- **Validierung:** Tab 4 nutzen um Modellgenauigkeit zu prüfen
- **Batteriespeicher:** E/P-Verhältnis 2-4 Stunden ist typisch für Netz-Batterien

### Daten-Export
- Ergebnisse werden (noch) nicht automatisch exportiert
- Verwende Browser-Screenshots für schnelle Dokumentation
- TODO: CSV/JSON-Export implementieren

## 🆘 Troubleshooting

### "No SMARD files found"
→ Führe `python import_smard_data.py` aus, um SMARD-Daten zu importieren

### "Optimization failed"
→ Prüfe, ob CBC-Solver installiert ist: `cbc -v`
→ Reduziere Zeitraum (weniger Snapshots)

### "Could not load real prices"
→ SMARD-Preisdaten fehlen, nur Simulation möglich (keine Validierung)

### Dashboard lädt nicht
→ Prüfe, ob Streamlit installiert ist: `pip install streamlit`
→ Starte mit: `streamlit run dashboard/app.py`

## 📚 Weiterführende Dokumentation

- [PyPSA Documentation](https://pypsa.readthedocs.io/)
- [SMARD Data Portal](https://www.smard.de/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

**Version:** 1.0
**Erstellt:** 2025-02
**Framework:** PyPSA + Streamlit
