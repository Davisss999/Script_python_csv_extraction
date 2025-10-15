# Script Modulare per Analisi Dati Anthem

## Panoramica delle Modifiche

Lo script è stato reso **completamente modulare** e ora processa automaticamente tutti i file CSV presenti nella cartella specificata.

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
- **Grafici combinati** per ogni tipo di dispositivo (EM, RM, SM)
- **Forme d'onda pulse rate** per dispositivi SM
- **Mappa interattiva** con timestamp univoco
- **Report di dispositivi** trovati per ogni file

## Teoria della Forma d'onda Pulse Rate

### Principio Fisico
La **fotopletismografia (PPG)** rileva variazioni volumetriche del sangue nei tessuti:

1. **LED Verde** → Illumina la pelle
2. **Fotodiodo** → Rileva luce riflessa
3. **Variazioni** → Il sangue assorbe più luce verde durante la pulsazione
4. **Segnale** → Variazioni periodiche nell'intensità luminosa

### Elaborazione del Segnale

#### Fase 1: Rimozione DC
```python
green_ac = green_clean - np.mean(green_clean)
```
- **Scopo**: Elimina l'offset costante dei tessuti statici
- **Risultato**: Mantiene solo le variazioni dinamiche

#### Fase 2: Filtro Passa-Banda (0.5-4 Hz)
```python
b, a = signal.butter(3, [low, high], btype='band')
waveform = signal.filtfilt(b, a, green_ac)
```
- **Frequenze**: 0.5-4 Hz = 30-240 bpm (range fisiologico)
- **Scopo**: Elimina rumore ad alta frequenza e derive a bassa frequenza
- **Metodo**: Filtro Butterworth bidirezionale (zero-phase)

#### Fase 3: Normalizzazione
```python
waveform_norm = (waveform - waveform.min()) / (waveform.max() - waveform.min())
```
- **Scopo**: Standardizza l'ampiezza per confronti tra soggetti
- **Range**: 0-1 normalizzato

### Interpretazione della Forma d'onda

1. **Picchi**: Corrispondono alla sistole (contrazione cardiaca)
2. **Valli**: Corrispondono alla diastole (rilassamento cardiaco)
3. **Frequenza**: Determina la frequenza cardiaca
4. **Morfologia**: Indica qualità del segnale e perfusione tissutale

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