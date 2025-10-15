# Script Python CSV Extraction

Script Python modulare per l'estrazione, analisi e visualizzazione di dati da dispositivi di monitoraggio Anthem.

## 📋 Descrizione

Questo progetto contiene script Python per processare dati da tre tipi di dispositivi:
- **Environmental Monitor (EM)**: CO2, temperatura, umidità, VOC, PM2.5, PM10, ecc.
- **Respiratory Monitor (RM)**: sensori IMU (accelerometro, giroscopio, magnetometro)
- **Stress Monitor (SM)**: frequenza cardiaca, sensori PPG (fotopletismografia)

## 🚀 Caratteristiche

### Script Principale: `lettura_modulare.py`
- ✅ **Architettura modulare** per gestire più dispositivi contemporaneamente
- ✅ **Parsing automatico** con rilevamento del tipo di dispositivo
- ✅ **Grafici interattivi** con matplotlib
- ✅ **Mappa interattiva** con folium e layer multipli
- ✅ **Estrazione forma d'onda pulse rate** da segnale PPG
- ✅ **Supporto multi-dispositivo** dello stesso tipo con overlay colorati

### Funzionalità Principali

#### Environmental Monitor (EM)
- Visualizzazione di tutte le metriche ambientali
- Griglia combinata con subplot per ogni metrica
- Layer mappa con gradiente CO2 (blu → rosso)

#### Respiratory Monitor (RM)
- Grafici per accelerometro, giroscopio e magnetometro
- Calcolo magnitudine movimento
- Layer mappa con gradiente movimento (verde → giallo)

#### Stress Monitor (SM)
- Grafici HR e HR-Confidence
- Sensori PPG (Red, IR, Green)
- **Estrazione forma d'onda pulse rate** con filtraggio passa-banda
- Layer mappa con gradiente HR (viola → rosso)

## 📦 Requisiti

```bash
pip install folium branca matplotlib numpy scipy
```

## 🔧 Installazione

1. Clona il repository:
```bash
git clone https://github.com/[USERNAME]/Script_python_csv_extraction.git
cd Script_python_csv_extraction
```

2. Crea un ambiente virtuale (opzionale ma consigliato):
```bash
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
```

3. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

## 📊 Utilizzo

### Script Modulare

```python
# Modifica il percorso del CSV nel file lettura_modulare.py
csv_path = r"C:\path\to\your\Unico.csv"

# Esegui lo script
python lettura_modulare.py
```

### Formato CSV

Il CSV deve avere le seguenti colonne:
- `TIMESTAMP`: formato ISO 8601
- `GPS_COORDINATES`: lat,lon
- `DEVICE_TYPE`: EM, RM, o SM
- `DEVICE_NAME`: identificativo del dispositivo
- `DATA received`: dati specifici del dispositivo

### Output

Lo script genera:
1. **Grafici matplotlib** per ogni tipo di dispositivo
2. **Mappa interattiva HTML** (`anthem_devices_map.html`)
3. **Grafico forma d'onda pulse rate** per SM

## 📁 Struttura File

```
Script_python_csv_extraction/
├── lettura_modulare.py       # Script principale modulare
├── lettura2.py                # Script originale per EM
├── CSV_Python/                # Directory per file CSV
│   ├── EM.csv
│   ├── RM.csv
│   ├── SM.csv
│   └── Unico.csv
├── requirements.txt           # Dipendenze Python
└── README.md                  # Questo file
```

## 🔬 Dettagli Tecnici

### Pulse Rate Waveform Extraction
- Rimozione componente DC
- Filtro passa-banda Butterworth (0.5-4 Hz = 30-240 bpm)
- Normalizzazione per visualizzazione ottimale

### Processing Pipeline
1. Lettura CSV con parser personalizzato
2. Routing al parser specifico per device type
3. Ordinamento temporale
4. Creazione serie temporali allineate
5. Visualizzazione separata per tipo dispositivo

## 📝 Note

- Lo script gestisce automaticamente dispositivi multipli dello stesso tipo
- I grafici mostrano overlay con colori diversi per dispositivi dello stesso tipo
- La mappa include layer attivabili/disattivabili per ogni metrica principale

## 🤝 Contributi

Contributi, issues e feature requests sono benvenuti!

## 📄 Licenza

Questo progetto è distribuito sotto licenza MIT.

## 👤 Autore

Sviluppato per l'analisi di dati da dispositivi Anthem.
