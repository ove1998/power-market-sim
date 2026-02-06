# SMARD Daten Download-Anleitung

## 🌐 SMARD.de Downloadcenter

**URL:** https://www.smard.de/home/downloadcenter/download-marktdaten

---

## 📊 Benötigte Dateien für 2025 (Ganzes Jahr)

Gehe zu SMARD Downloadcenter und lade folgende CSV-Dateien herunter:

### **1. Stromverbrauch (Demand)**
- **Kategorie:** Stromverbrauch
- **Datentyp:** Realisierter Stromverbrauch - Gesamt (Netzlast)
- **Zeitraum:** 01.01.2025 - 31.12.2025
- **Auflösung:** Stündlich
- **Dateiname:** `Realisierter_Stromverbrauch_202501010000_202512312359_Stunde.csv`
- **Speichern als:** `data/raw/smard/demand_2025.csv`

### **2. Stromerzeugung - Wind Onshore**
- **Kategorie:** Stromerzeugung
- **Datentyp:** Wind Onshore
- **Zeitraum:** 01.01.2025 - 31.12.2025
- **Auflösung:** Stündlich
- **Dateiname:** `Wind_Onshore_202501010000_202512312359_Stunde.csv`
- **Speichern als:** `data/raw/smard/wind_onshore_2025.csv`

### **3. Stromerzeugung - Wind Offshore**
- **Kategorie:** Stromerzeugung
- **Datentyp:** Wind Offshore
- **Zeitraum:** 01.01.2025 - 31.12.2025
- **Auflösung:** Stündlich
- **Dateiname:** `Wind_Offshore_202501010000_202512312359_Stunde.csv`
- **Speichern als:** `data/raw/smard/wind_offshore_2025.csv`

### **4. Stromerzeugung - Solar**
- **Kategorie:** Stromerzeugung
- **Datentyp:** Photovoltaik
- **Zeitraum:** 01.01.2025 - 31.12.2025
- **Auflösung:** Stündlich
- **Dateiname:** `Photovoltaik_202501010000_202512312359_Stunde.csv`
- **Speichern als:** `data/raw/smard/solar_2025.csv`

### **5. Stromerzeugung - Kernenergie**
- **Kategorie:** Stromerzeugung
- **Datentyp:** Kernenergie
- **Zeitraum:** 01.01.2025 - 31.12.2025
- **Auflösung:** Stündlich
- **Speichern als:** `data/raw/smard/nuclear_2025.csv`

### **6. Stromerzeugung - Braunkohle**
- **Kategorie:** Stromerzeugung
- **Datentyp:** Braunkohle
- **Zeitraum:** 01.01.2025 - 31.12.2025
- **Auflösung:** Stündlich
- **Speichern als:** `data/raw/smard/lignite_2025.csv`

### **7. Stromerzeugung - Steinkohle**
- **Kategorie:** Stromerzeugung
- **Datentyp:** Steinkohle
- **Zeitraum:** 01.01.2025 - 31.12.2025
- **Auflösung:** Stündlich
- **Speichern als:** `data/raw/smard/hard_coal_2025.csv`

### **8. Stromerzeugung - Erdgas**
- **Kategorie:** Stromerzeugung
- **Datentyp:** Erdgas
- **Zeitraum:** 01.01.2025 - 31.12.2025
- **Auflösung:** Stündlich
- **Speichern als:** `data/raw/smard/gas_2025.csv`
- **Hinweis:** SMARD unterscheidet nicht zwischen CCGT und OCGT - wir schätzen das später

### **9. Stromerzeugung - Wasserkraft**
- **Kategorie:** Stromerzeugung
- **Datentyp:** Wasserkraft
- **Zeitraum:** 01.01.2025 - 31.12.2025
- **Auflösung:** Stündlich
- **Speichern als:** `data/raw/smard/hydro_2025.csv`
- **Hinweis:** SMARD hat meist Gesamt-Wasserkraft (Run-of-River + Pumpspeicher)

### **10. Stromerzeugung - Biomasse**
- **Kategorie:** Stromerzeugung
- **Datentyp:** Biomasse
- **Zeitraum:** 01.01.2025 - 31.12.2025
- **Auflösung:** Stündlich
- **Speichern als:** `data/raw/smard/biomass_2025.csv`

### **11. Pumpspeicher**
- **Kategorie:** Stromerzeugung
- **Datentyp:** Pumpspeicher
- **Zeitraum:** 01.01.2025 - 31.12.2025
- **Auflösung:** Stündlich
- **Speichern als:** `data/raw/smard/pumped_hydro_2025.csv`

### **12. Grenzüberschreitender Stromhandel (Import/Export)**
- **Kategorie:** Grenzüberschreitender Handel
- **Datentyp:** Physikalische Stromflüsse - Gesamt
- **Zeitraum:** 01.01.2025 - 31.12.2025
- **Auflösung:** Stündlich
- **Speichern als:** `data/raw/smard/cross_border_2025.csv`
- **Wichtig:** Positiv = Import, Negativ = Export

### **13. Großhandelspreise (Day-Ahead)**
- **Kategorie:** Marktdaten
- **Datentyp:** Day-ahead Auktion
- **Zeitraum:** 01.01.2025 - 31.12.2025
- **Auflösung:** Stündlich
- **Dateiname:** `Day-ahead_Auktion_202501010000_202512312359_Stunde.csv`
- **Speichern als:** `data/raw/smard/day_ahead_prices_2025.csv`

### **14. Installierte Leistung (Stand 2025)**
- **Kategorie:** Installierte Erzeugungsleistung
- **Datentyp:** Alle Technologien
- **Zeitraum:** Aktueller Monat (z.B. Dezember 2025)
- **Auflösung:** Monatlich
- **Speichern als:** `data/raw/smard/installed_capacity_2025.csv`

---

## 📁 Dateistruktur

Nach dem Download sollte die Struktur so aussehen:

```
power-market-sim/
├── data/
│   ├── raw/
│   │   └── smard/
│   │       ├── demand_2025.csv
│   │       ├── wind_onshore_2025.csv
│   │       ├── wind_offshore_2025.csv
│   │       ├── solar_2025.csv
│   │       ├── nuclear_2025.csv
│   │       ├── lignite_2025.csv
│   │       ├── hard_coal_2025.csv
│   │       ├── gas_2025.csv
│   │       ├── hydro_2025.csv
│   │       ├── biomass_2025.csv
│   │       ├── pumped_hydro_2025.csv
│   │       ├── cross_border_2025.csv
│   │       ├── day_ahead_prices_2025.csv
│   │       └── installed_capacity_2025.csv
│   └── processed/
│       └── (wird automatisch generiert)
```

---

## ⚙️ CSV-Format (Beispiel)

SMARD CSV-Dateien haben typischerweise folgendes Format:

```csv
Datum;Uhrzeit;Gesamt (Netzlast) [MWh] Originalauflösungen
01.01.2025;00:00;45678,0
01.01.2025;01:00;43234,0
...
```

**Wichtig:**
- Semikolon (`;`) als Trennzeichen
- Komma (`,`) als Dezimaltrennzeichen
- Deutsche Datumsformat (DD.MM.YYYY)
- Zeitzone: MEZ/MESZ (Europa/Berlin)

---

## 🚀 Nach dem Download

Führe den Import aus:

```bash
python src/data_processing/smard_importer.py
```

Das Skript wird:
1. CSV-Dateien einlesen
2. Zeitstempel normalisieren (UTC)
3. Kapazitätsfaktoren berechnen (aus Erzeugung / installierte Leistung)
4. Parquet-Dateien speichern (schneller & kompakter)
5. Validierung durchführen

---

## 📞 Support

Bei Problemen mit dem SMARD Download:
- **SMARD Support:** https://www.smard.de/home/kontakt
- **Dokumentation:** https://www.smard.de/home/wiki-article/444/444

---

## ✅ Checkliste

**Erneuerbare Energien:**
- [ ] Wind Onshore Erzeugung 2025
- [ ] Wind Offshore Erzeugung 2025
- [ ] Solar (Photovoltaik) Erzeugung 2025
- [ ] Wasserkraft Erzeugung 2025
- [ ] Biomasse Erzeugung 2025

**Konventionelle Kraftwerke:**
- [ ] Kernenergie Erzeugung 2025
- [ ] Braunkohle Erzeugung 2025
- [ ] Steinkohle Erzeugung 2025
- [ ] Erdgas Erzeugung 2025

**Speicher & Handel:**
- [ ] Pumpspeicher Erzeugung 2025
- [ ] Grenzüberschreitender Handel (Import/Export) 2025

**Nachfrage & Preise:**
- [ ] Demand (Stromverbrauch) 2025
- [ ] Day-Ahead Preise 2025

**Kapazitäten:**
- [ ] Installierte Leistung pro Technologie 2025

**Import & Verarbeitung:**
- [ ] Alle Dateien in `data/raw/smard/` gespeichert
- [ ] Import-Skript ausgeführt
- [ ] Validierung durchgeführt
