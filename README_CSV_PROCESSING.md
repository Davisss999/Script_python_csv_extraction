# Script Modulare per Analisi Dati Anthem

## Panoramica delle Modifiche

Lo script è stato reso **completamente modulare** e ora processa automaticamente tutti i file CSV presenti nella cartella specificata.

## Dispositivi Supportati

### 🌍 Environmental Monitor (EM)
- **Formato**: `CO2: 1440ppm, CO: 24.61ppm, PM2.5: 8.5μg/m³, ...`
- **Metriche**: 16 parametri ambientali
- **Grafici**: Griglia 2x8 con statistiche per ogni metrica

### 🫁 Respiratory Monitor (RM)  
- **Formato**: `Accel: (x,y,z), Gyro: (x,y,z), Mag: (x,y,z)`
- **Metriche**: Accelerometro, Giroscopio, Magnetometro + Magnitudine
- **Grafici**: 4 subplot con sensori IMU

### ❤️ Polar H10 (PH) - **NUOVO!**
- **Formato**: `HR: 72bpm, Contact: N/A, RR: [776]ms`
- **Metriche**: 
  - **HR**: Heart Rate (frequenza cardiaca in bpm)
  - **Contact**: Stato contatto sensore
  - **RR**: Intervalli R-R in millisecondi (HRV)
  - **RR_Mean**: Media intervalli multipli
  - **RR_Count**: Numero di intervalli
- **Grafici**: 
  - Heart Rate con statistiche
  - RR Intervals (HRV)
  - Contact Status
- **Analisi HRV**:
  - SDNN (variabilità globale)
  - RMSSD (variabilità a breve termine)
  - pNN50 (percentuale differenze >50ms)
  - Poincaré Plot
  - Distribuzione RR intervals

## Come Usare lo Script

### 1. Preparazione
- Assicurati che tutti i file CSV siano nella cartella: `C:\Users\david\Desktop portatile\CSV_Python`
- Non è più necessario modificare il codice per ogni nuovo file

### 2. Esecuzione
```bash
python lettura_modulare.py
```

### 3. Risultati
Lo script genererà automaticamente:
- **Grafici separati** per ogni dispositivo (EM, RM, PH)
- **Analisi HRV** per dispositivi Polar H10
- **Mappa interattiva** con layer per ogni dispositivo
- **Report dettagliato** in cartella timestampata

## Teoria dell'Analisi HRV (Heart Rate Variability)

### Principio Fisiologico
L'**HRV** misura la variazione temporale tra battiti cardiaci consecutivi. È un indicatore importante di:
- Attività del sistema nervoso autonomo
- Stress fisiologico e psicologico
- Fitness cardiovascolare
- Recupero dopo attività fisica

### Metriche HRV Calcolate

#### 1. SDNN (Standard Deviation of NN intervals)
```python
sdnn = np.std(rr_intervals)
```
- **Cosa misura**: Variabilità globale degli intervalli RR
- **Interpretazione**: 
  - SDNN elevato → Buona variabilità, sistema autonomo sano
  - SDNN basso → Ridotta variabilità, possibile stress o affaticamento

#### 2. RMSSD (Root Mean Square of Successive Differences)
```python
rr_diffs = np.diff(rr_intervals)
rmssd = np.sqrt(np.mean(rr_diffs ** 2))
```
- **Cosa misura**: Variabilità a breve termine (parasimpatico)
- **Interpretazione**:
  - RMSSD alto → Buona attività parasimpatica, rilassamento
  - RMSSD basso → Stress, attivazione simpatica

#### 3. pNN50 (percentage of successive NN intervals > 50ms)
```python
nn50 = np.sum(np.abs(rr_diffs) > 50)
pnn50 = (nn50 / len(rr_diffs)) * 100
```
- **Cosa misura**: Percentuale di intervalli con differenza >50ms
- **Interpretazione**:
  - pNN50 alto (>20%) → Buona modulazione parasimpatica
  - pNN50 basso (<5%) → Ridotta variabilità, possibile stress

### Visualizzazioni HRV

1. **Time Series RR**: Mostra andamento intervalli RR nel tempo
2. **Poincaré Plot**: Visualizza pattern HRV (RR[n] vs RR[n+1])
3. **Istogramma**: Distribuzione degli intervalli RR

## Funzionalità Aggiuntive

### Gestione Multi-File
- **Nomi univoci**: Ogni dispositivo include il nome del file sorgente
- **Aggregazione**: Combina dati da file multipli
- **Tracciabilità**: Mantiene traccia del file di origine

### Visualizzazioni Migliorate
- **Legenda chiara**: Mostra dispositivo + file sorgente
- **Colori distintivi**: Diversi colori per dispositivi multipli
- **Timestamp**: File mappa con timestamp per evitare sovrascritture

### Output di Log
```
🔄 Processando: file1.csv
   ✅ Righe processate: 1250, saltate: 15

📊 File processati: 3
📈 Grafici generati per 3 tipi di dispositivi
   - Environmental Monitor: 2 dispositivi
   - Respiratory Monitor: 1 dispositivi  
   - Stress Monitor: 2 dispositivi
```

## Struttura Dati

```python
data_by_device[device_type][device_name_unique] = {
    'timestamps_raw': [],      # Timestamp originali
    'timestamps_dt': [],       # Datetime objects ordinati
    'gps_coords': [],         # Coordinate (lat, lon)
    'row_metrics': [],        # Metriche per ogni campione
    'source_file': '',        # File CSV di origine
    'series': {},             # Serie temporali per grafici
    'metric_order': []        # Ordine metriche
}
```

## Aggiungere Nuovi File

1. **Copia** il nuovo file CSV nella cartella `CSV_Python`
2. **Esegui** lo script - rileverà automaticamente il nuovo file
3. **Visualizza** i risultati aggregati con tutti i dispositivi

Non sono necessarie modifiche al codice!