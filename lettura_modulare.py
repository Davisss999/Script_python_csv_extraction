# Script modulare per leggere e visualizzare dati da dispositivi Anthem
# Supporta: Environmental Monitor (EM), Respiratory Monitor (RM), Stress Monitor (SM)
# Processamento automatico di tutti i file CSV nella cartella specificata

import csv
import math
import os
import re
import webbrowser
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import branca
import folium
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import signal

# Configura backend matplotlib per Windows
try:
    # Prova diversi backend in ordine di preferenza
    backends_to_try = ['Qt5Agg', 'TkAgg', 'Agg']
    for backend in backends_to_try:
        try:
            matplotlib.use(backend)
            print(f"🖼️  Usando backend matplotlib: {backend}")
            break
        except ImportError:
            continue
    else:
        # Se nessun backend interattivo funziona, usa Agg per salvare i grafici
        matplotlib.use('Agg')
        print("🖼️  Usando backend Agg - i grafici saranno salvati come immagini")
except Exception as e:
    print(f"⚠️  Problema con backend matplotlib: {e}")
    matplotlib.use('Agg')

# =====================================================================
# CONFIGURAZIONE
# =====================================================================

# Cartella contenente i file CSV da processare
csv_directory = r"C:\Users\david\Desktop portatile\CSV_Python"

# Cartella di output per i risultati
output_base_directory = r"C:\Users\david\Desktop portatile\CSV_Python\output"

# --- Righe da saltare ---
SKIP_ROWS = set()  # set(range(1, 101)) se vuoi saltare righe
SKIP_TIMESTAMPS = set()

def should_skip(row_idx: int, row: dict) -> bool:
    return False

# --- Formato timestamp ---
DISPLAY_TS_FMT = '%d/%m/%Y %H:%M:%S'

# =====================================================================
# FUNZIONI DI UTILITÀ PER GESTIONE MULTIPLA CSV
# =====================================================================

def find_csv_files(directory: str) -> list:
    """Trova tutti i file CSV nella directory specificata"""
    csv_dir = Path(directory)
    if not csv_dir.exists():
        print(f"⚠️  Directory non trovata: {directory}")
        return []
    
    csv_files = list(csv_dir.glob("*.csv"))
    print(f"📁 Trovati {len(csv_files)} file CSV in {directory}")
    for csv_file in csv_files:
        print(f"   - {csv_file.name}")
    
    return [str(f) for f in csv_files]

def process_single_csv(csv_path: str, data_by_device: dict, units_seen: dict) -> int:
    """Processa un singolo file CSV e aggiorna le strutture dati globali"""
    print(f"\n🔄 Processando: {Path(csv_path).name}")
    
    skipped_count = 0
    processed_rows = 0
    
    try:
        with open(csv_path, newline='', encoding='utf-8-sig') as f:
            # Usa il reader CSV standard per gestire meglio le virgolette
            csv_reader = csv.reader(f)
            
            # Leggi header
            try:
                headers = next(csv_reader)
            except StopIteration:
                print(f"   ⚠️ File vuoto: {csv_path}")
                return 0
            
            for row_idx, values in enumerate(csv_reader, start=1):
                if len(values) != len(headers):
                    continue
                
                # Crea dizionario row
                row = dict(zip(headers, values))
                
                # Salta righe configurate
                raw_ts = (row.get('TIMESTAMP') or '').strip()
                if (row_idx in SKIP_ROWS) or (raw_ts in SKIP_TIMESTAMPS) or should_skip(row_idx, row):
                    skipped_count += 1
                    continue
                
                # Estrai campi base
                ts = raw_ts
                gps = (row.get('GPS_COORDINATES') or '').strip()
                device_type = (row.get('DEVICE_TYPE') or '').strip()
                device_name = (row.get('DEVICE_NAME') or row.get('DEVICE') or '').strip()
                data_str = (row.get('DATA received') or '').strip()
                
                if not ts or not gps or not device_type or not data_str:
                    continue
                
                # Parse GPS
                try:
                    lat_s, lon_s = [s.strip() for s in gps.split(',', 1)]
                    lat, lon = float(lat_s), float(lon_s)
                except Exception:
                    continue
                
                # Routing al parser appropriato
                if device_type == 'EM':
                    metrics_dict = parse_EM(data_str)
                elif device_type == 'RM':
                    metrics_dict = parse_RM(data_str)
                elif device_type == 'SM':
                    metrics_dict = parse_SM(data_str)
                else:
                    print(f"⚠️  Device type sconosciuto: {device_type}")
                    continue
                
                if not metrics_dict:
                    continue
                
                # Crea chiave univoca per dispositivo che include il file sorgente
                file_prefix = Path(csv_path).stem
                unique_device_name = f"{device_name}_{file_prefix}" if device_name else file_prefix
                
                # Salva dati per dispositivo
                device_data = data_by_device[device_type][unique_device_name]
                device_data['timestamps_raw'].append(ts)
                device_data['gps_coords'].append((lat, lon))
                device_data['row_metrics'].append(metrics_dict)
                device_data['source_file'] = csv_path  # Traccia file sorgente
                
                # Traccia unità (solo EM per ora)
                if device_type == 'EM':
                    for name, num, unit in PATTERN_EM.findall(data_str):
                        norm = normalize_metric_name(name)
                        unit = unit.strip()
                        if unit and norm not in units_seen:
                            units_seen[norm] = unit
                
                processed_rows += 1
    
    except Exception as e:
        print(f"❌ Errore processando {csv_path}: {e}")
        return 0
    
    print(f"   ✅ Righe processate: {processed_rows}, saltate: {skipped_count}")
    return processed_rows

# =====================================================================
# PARSER SPECIALIZZATI PER DEVICE TYPE
# =====================================================================

def parse_ts(ts):
    """Parse timestamp ISO format"""
    t = ts.strip().replace('Z', '')
    try:
        return datetime.fromisoformat(t)
    except ValueError:
        return datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f')

def normalize_metric_name(name: str) -> str:
    """Normalizza nomi metriche (per EM)"""
    h = re.sub(r'[\u2010-\u2015\u2212]', '-', name.strip())
    h = re.sub(r'\s*-\s*', '-', h)
    h = re.sub(r'\s+', ' ', h)
    no_space_up = re.sub(r'\s+', '', h).upper()
    if re.fullmatch(r'VOC-?I(NDEX)?', no_space_up):
        return 'VOC index'
    if re.fullmatch(r'NOX-?I(NDEX)?', no_space_up) or re.fullmatch(r'NOXINDEX', no_space_up):
        return 'NOx index'
    return h

# --- Parser EM: "Nome: valore [unità], ..." ---
PATTERN_EM = re.compile(r'\s*([^:,]+?)\s*:\s*([-+]?\d*\.?\d+)\s*([^,]*)')

def parse_EM(data_str: str) -> dict:
    """Parse Environmental Monitor data"""
    metrics = {}
    for name, num, unit in PATTERN_EM.findall(data_str):
        norm = normalize_metric_name(name)
        try:
            val = float(num)
            metrics[norm] = val
        except ValueError:
            continue
    return metrics

# --- Parser RM: "Accel: (x,y,z), Gyro: (x,y,z), Mag: (x,y,z)" ---
PATTERN_RM = re.compile(r'(Accel|Gyro|Mag):\s*\(([^)]+)\)')

def parse_RM(data_str: str) -> dict:
    """Parse Respiratory Monitor data (IMU sensors)"""
    metrics = {}
    for sensor, values_str in PATTERN_RM.findall(data_str):
        try:
            vals = [float(v.strip()) for v in values_str.split(',')]
            if len(vals) == 3:
                metrics[f'{sensor}_X'] = vals[0]
                metrics[f'{sensor}_Y'] = vals[1]
                metrics[f'{sensor}_Z'] = vals[2]
        except ValueError:
            continue
    
    # Calcola magnitudine accelerazione (per mappa)
    if 'Accel_X' in metrics and 'Accel_Y' in metrics and 'Accel_Z' in metrics:
        x, y, z = metrics['Accel_X'], metrics['Accel_Y'], metrics['Accel_Z']
        metrics['Accel_Magnitude'] = math.sqrt(x**2 + y**2 + z**2)
    
    return metrics

# --- Parser SM: "HR: 49.0bpm, HR-Conf: 100.0%, ..." ---
PATTERN_SM = re.compile(r'([^:,]+?):\s*([-+]?\d*\.?\d+)\s*([a-zA-Z%]*)')

def parse_SM(data_str: str) -> dict:
    """Parse Stress Monitor data"""
    metrics = {}
    for name, num, unit in PATTERN_SM.findall(data_str):
        name = name.strip()
        try:
            val = float(num)
            metrics[name] = val
        except ValueError:
            continue
    return metrics

# =====================================================================
# LETTURA E ORGANIZZAZIONE DATI
# =====================================================================

def parse_csv_line(line):
    """Parse una riga CSV usando il parser standard di Python"""
    # Usa il parser CSV standard che gestisce meglio le virgolette
    import io
    reader = csv.reader([line])
    try:
        return next(reader)
    except StopIteration:
        return []

# Struttura: data_by_device[device_type][device_name] = {'timestamps': [], 'gps': [], 'metrics': {}}
data_by_device = defaultdict(lambda: defaultdict(lambda: {
    'timestamps_raw': [],
    'timestamps_dt': [],
    'gps_coords': [],
    'row_metrics': [],
    'source_file': ''
}))

units_seen = {}  # Traccia unità per ogni metrica

print("🔄 Lettura CSV multipli...")

# Trova e processa tutti i file CSV nella directory
csv_files = find_csv_files(csv_directory)

if not csv_files:
    print("❌ Nessun file CSV trovato nella directory specificata!")
    exit(1)

total_processed = 0
for csv_file in csv_files:
    processed = process_single_csv(csv_file, data_by_device, units_seen)
    total_processed += processed

print(f"\n✅ Totale righe processate: {total_processed}")
print(f"✅ Dispositivi trovati:")
for dtype, devices in data_by_device.items():
    for dname, ddata in devices.items():
        source_file = Path(ddata['source_file']).name if ddata['source_file'] else 'N/A'
        print(f"   - {dtype}/{dname}: {len(ddata['timestamps_raw'])} campioni (da {source_file})")

# =====================================================================
# ORDINAMENTO TEMPORALE PER OGNI DISPOSITIVO
# =====================================================================

print("\n🔄 Ordinamento temporale...")

for device_type, devices in data_by_device.items():
    for device_name, device_data in devices.items():
        # Parse timestamps
        timestamps_dt = [parse_ts(ts) for ts in device_data['timestamps_raw']]
        
        # Ordina
        order = sorted(range(len(timestamps_dt)), key=lambda i: timestamps_dt[i])
        device_data['timestamps_dt'] = [timestamps_dt[i] for i in order]
        device_data['timestamps_raw'] = [device_data['timestamps_raw'][i] for i in order]
        device_data['gps_coords'] = [device_data['gps_coords'][i] for i in order]
        device_data['row_metrics'] = [device_data['row_metrics'][i] for i in order]

# =====================================================================
# PREPARAZIONE SERIE TEMPORALI
# =====================================================================

print("🔄 Preparazione serie temporali...")

for device_type, devices in data_by_device.items():
    for device_name, device_data in devices.items():
        # Estrai tutte le metriche uniche
        metric_order = []
        seen = set()
        for d in device_data['row_metrics']:
            for k in d.keys():
                if k not in seen:
                    seen.add(k)
                    metric_order.append(k)
        
        # Crea serie allineate
        series = {m: [] for m in metric_order}
        for d in device_data['row_metrics']:
            for m in metric_order:
                series[m].append(d.get(m, np.nan))
        
        device_data['metric_order'] = metric_order
        device_data['series'] = series
        
        print(f"   {device_type}/{device_name}: {len(metric_order)} metriche")

# =====================================================================
# VISUALIZZAZIONE GRAFICI PER DEVICE TYPE
# =====================================================================

print("\n📊 Generazione grafici...")

def _format_time_axis(ax):
    """Formatta asse tempo"""
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(DISPLAY_TS_FMT))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha('right')

def _get_device_display_name(device_name: str) -> str:
    """Estrae nome dispositivo pulito per visualizzazione"""
    # Rimuove suffisso del file se presente
    if '_' in device_name:
        parts = device_name.split('_')
        if len(parts) > 1:
            # Usa solo la parte prima dell'underscore se sembra un nome file
            base_name = parts[0]
            file_part = '_'.join(parts[1:])
            return f"{base_name} ({file_part})"
    return device_name

def create_output_directory(timestamp_str: str) -> str:
    """Crea la directory di output con timestamp"""
    output_dir = Path(output_base_directory) / f"analysis_{timestamp_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Directory di output creata: {output_dir}")
    return str(output_dir)

def _save_figure(fig, filepath: str):
    """Salva una figura come immagine PNG"""
    try:
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   💾 Grafico salvato: {Path(filepath).name}")
        return filepath
    except Exception as e:
        print(f"   ⚠️  Errore salvando {Path(filepath).name}: {e}")
        return None

# Colori per dispositivi multipli dello stesso tipo
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# Crea directory di output con timestamp
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = create_output_directory(timestamp_str)

# --- GRAFICI EM ---
if 'EM' in data_by_device:
    print("   📈 Grafici Environmental Monitor...")
    
    # Crea un grafico separato per ogni dispositivo EM
    for device_name, device_data in data_by_device['EM'].items():
        display_name = _get_device_display_name(device_name)
        metrics = device_data['metric_order']
        
        if not metrics:
            continue
            
        print(f"      📊 Generando grafico per: {display_name}")
        
        # Griglia per le metriche del dispositivo
        ncols = 2
        nrows = math.ceil(len(metrics) / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(16, max(8, nrows * 4)), 
                               sharex='all', squeeze=False)
        fig.suptitle(f'Environmental Monitor - {display_name}', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            r, c = divmod(i, ncols)
            ax = axs[r, c]
            
            # Plot del singolo dispositivo
            ax.plot(device_data['timestamps_dt'], device_data['series'][metric], 
                   color='#1f77b4', linewidth=2, alpha=0.8)
            
            unit = (units_seen.get(metric, '') or '').strip()
            ylabel = f"{metric} ({unit})" if unit else metric
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(metric, fontsize=12, pad=10)
            _format_time_axis(ax)
            ax.grid(True, alpha=0.3)
            
            # Aggiungi statistiche
            values = np.array(device_data['series'][metric])
            finite_values = values[np.isfinite(values)]
            if len(finite_values) > 0:
                mean_val = np.mean(finite_values)
                std_val = np.std(finite_values)
                ax.text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=10)
        
        # Nascondi subplot in eccesso
        for j in range(len(metrics), nrows * ncols):
            r, c = divmod(j, ncols)
            axs[r, c].set_visible(False)
        
        plt.tight_layout()
        
        # Salva il grafico del dispositivo
        safe_name = device_name.replace('/', '_').replace(' ', '_')
        filepath = Path(output_dir) / f"EM_{safe_name}.png"
        _save_figure(fig, str(filepath))
        plt.close(fig)

# --- GRAFICI RM ---
if 'RM' in data_by_device:
    print("   📈 Grafici Respiratory Monitor...")
    
    # Crea un grafico separato per ogni dispositivo RM
    for device_name, device_data in data_by_device['RM'].items():
        display_name = _get_device_display_name(device_name)
        
        print(f"      📊 Generando grafico per: {display_name}")
        
        # Crea griglia per Accel, Gyro, Mag + Magnitudine
        fig, axs = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        fig.suptitle(f'Respiratory Monitor - {display_name}', fontsize=16, fontweight='bold')
        
        sensor_groups = [
            (['Accel_X', 'Accel_Y', 'Accel_Z'], 'Accelerometer (g)', 0, '#e74c3c'),
            (['Gyro_X', 'Gyro_Y', 'Gyro_Z'], 'Gyroscope (°/s)', 1, '#3498db'),
            (['Mag_X', 'Mag_Y', 'Mag_Z'], 'Magnetometer (μT)', 2, '#2ecc71'),
            (['Accel_Magnitude'], 'Acceleration Magnitude (g)', 3, '#f39c12')
        ]
        
        for metrics_group, ylabel, ax_idx, base_color in sensor_groups:
            ax = axs[ax_idx]
            
            # Plot X, Y, Z con varianti di colore o magnitudine
            for axis_idx, metric in enumerate(metrics_group):
                if metric in device_data['series']:
                    if metric == 'Accel_Magnitude':
                        # Magnitudine con colore pieno
                        ax.plot(device_data['timestamps_dt'], device_data['series'][metric],
                               label=metric, color=base_color, linewidth=2)
                    else:
                        # X, Y, Z con linestyle diversi
                        alpha = 0.9 - axis_idx * 0.1
                        linestyle = ['-', '--', ':'][axis_idx]
                        axis_name = metric.split('_')[1]
                        ax.plot(device_data['timestamps_dt'], device_data['series'][metric],
                               label=f"{axis_name}", color=base_color, linestyle=linestyle, 
                               linewidth=2, alpha=alpha)
            
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(ylabel.split('(')[0].strip(), fontsize=12, pad=10)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        _format_time_axis(axs[-1])
        plt.tight_layout()
        
        # Salva il grafico del dispositivo
        safe_name = device_name.replace('/', '_').replace(' ', '_')
        filepath = Path(output_dir) / f"RM_{safe_name}.png"
        _save_figure(fig, str(filepath))
        plt.close(fig)

# --- GRAFICI SM ---
if 'SM' in data_by_device:
    print("   📈 Grafici Stress Monitor...")
    
    # Crea un grafico separato per ogni dispositivo SM
    for device_name, device_data in data_by_device['SM'].items():
        display_name = _get_device_display_name(device_name)
        
        print(f"      📊 Generando grafico per: {display_name}")
        
        # Crea griglia per HR/HR-Conf e PPG sensors
        fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        fig.suptitle(f'Stress Monitor - {display_name}', fontsize=16, fontweight='bold')
        
        # Subplot 1: HR e HR-Conf
        ax1 = axs[0]
        ax1_twin = ax1.twinx()
        
        if 'HR' in device_data['series']:
            ax1.plot(device_data['timestamps_dt'], device_data['series']['HR'],
                    label="Heart Rate", color='#e74c3c', linewidth=3)
            
            # Statistiche HR
            hr_values = np.array(device_data['series']['HR'])
            finite_hr = hr_values[np.isfinite(hr_values)]
            if len(finite_hr) > 0:
                mean_hr = np.mean(finite_hr)
                std_hr = np.std(finite_hr)
                ax1.text(0.02, 0.98, f'HR: μ={mean_hr:.1f} bpm\nσ={std_hr:.1f} bpm', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                        fontsize=11, fontweight='bold')
        
        if 'HR-Conf' in device_data['series']:
            ax1_twin.plot(device_data['timestamps_dt'], device_data['series']['HR-Conf'],
                         label="HR Confidence", color='#3498db', 
                         linestyle='--', linewidth=2, alpha=0.7)
        
        ax1.set_ylabel('Heart Rate (bpm)', color='#e74c3c', fontweight='bold', fontsize=12)
        ax1_twin.set_ylabel('HR Confidence (%)', color='#3498db', fontweight='bold', fontsize=12)
        ax1.set_title('Heart Rate Analysis', fontsize=14, pad=15)
        ax1.legend(loc='upper left', fontsize=10)
        ax1_twin.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Sensori PPG (Red, IR, Green)
        ax2 = axs[1]
        
        ppg_colors = {'Red': '#e74c3c', 'IR': '#34495e', 'Green': '#27ae60'}
        
        for ppg_name, ppg_color in ppg_colors.items():
            if ppg_name in device_data['series']:
                ax2.plot(device_data['timestamps_dt'], device_data['series'][ppg_name],
                        label=f"{ppg_name} Channel", color=ppg_color, 
                        linewidth=2, alpha=0.8)
        
        ax2.set_ylabel('PPG Sensors (ADC Counts)', fontweight='bold', fontsize=12)
        ax2.set_title('Photoplethysmography (PPG) Sensors', fontsize=14, pad=15)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        _format_time_axis(ax2)
        plt.tight_layout()
        
        # Salva il grafico del dispositivo
        safe_name = device_name.replace('/', '_').replace(' ', '_')
        filepath = Path(output_dir) / f"SM_{safe_name}.png"
        _save_figure(fig, str(filepath))
        plt.close(fig)
    
    # --- GRAFICO PULSE RATE WAVEFORM ---
    print("   📈 Estrazione forma d'onda Pulse Rate...")
    
    for device_name, device_data in data_by_device['SM'].items():
        if 'Green' not in device_data['series']:
            continue
        
        display_name = _get_device_display_name(device_name)
        print(f"   📈 Processando forma d'onda per: {display_name}")
        
        # Estrai dati Green channel
        green_data = np.array(device_data['series']['Green'], dtype=float)
        timestamps_dt = device_data['timestamps_dt']
        
        # Rimuovi NaN
        valid_mask = np.isfinite(green_data)
        green_clean = green_data[valid_mask]
        timestamps_clean = [timestamps_dt[i] for i, v in enumerate(valid_mask) if v]
        
        if len(green_clean) < 10:
            print(f"   ⚠️  Dati insufficienti per {device_name}")
            continue
        
        # Calcola frequenza di campionamento media
        if len(timestamps_clean) > 1:
            time_diffs = [(timestamps_clean[i+1] - timestamps_clean[i]).total_seconds() 
                         for i in range(len(timestamps_clean)-1) if i < len(timestamps_clean)-1]
            avg_sample_interval = np.mean(time_diffs)
            fs = 1.0 / avg_sample_interval if avg_sample_interval > 0 else 1.0
        else:
            fs = 1.0
        
        print(f"      Freq. campionamento: {fs:.2f} Hz")
        
        # 1. Rimuovi componente DC
        green_ac = green_clean - np.mean(green_clean)
        
        # 2. Filtro passa-banda (0.5-4 Hz = 30-240 bpm)
        nyquist = fs / 2
        low = 0.5 / nyquist
        high = 4.0 / nyquist
        
        # Limita i valori tra 0 e 1 (requisito del filtro butter)
        if low >= 1:
            low = 0.95
        if high >= 1:
            high = 0.95
        if low <= 0:
            low = 0.05
        
        try:
            b, a = signal.butter(3, [low, high], btype='band')
            waveform = signal.filtfilt(b, a, green_ac)
            
            # 3. Normalizza
            if waveform.max() != waveform.min():
                waveform_norm = (waveform - waveform.min()) / (waveform.max() - waveform.min())
            else:
                waveform_norm = waveform
            
            # Crea figura per la forma d'onda
            fig_wave, axs_wave = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            fig_wave.suptitle(f'Pulse Rate Waveform - {display_name}', fontsize=14, fontweight='bold')
            
            # Subplot 1: Segnale grezzo
            axs_wave[0].plot(timestamps_clean, green_clean, alpha=0.7, color='#27ae60', linewidth=1)
            axs_wave[0].set_ylabel('Green Channel\n(ADC Counts)', fontweight='bold')
            axs_wave[0].set_title('Segnale PPG Grezzo')
            axs_wave[0].grid(True, alpha=0.3)
            
            # Subplot 2: Segnale AC (DC removed)
            axs_wave[1].plot(timestamps_clean, green_ac, alpha=0.8, color='#3498db', linewidth=1)
            axs_wave[1].set_ylabel('Ampiezza AC', fontweight='bold')
            axs_wave[1].set_title('Segnale AC (DC Removed)')
            axs_wave[1].grid(True, alpha=0.3)
            axs_wave[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            
            # Subplot 3: Forma d'onda filtrata
            axs_wave[2].plot(timestamps_clean, waveform_norm, color='#e74c3c', linewidth=1.5)
            axs_wave[2].set_ylabel('Ampiezza\nNormalizzata', fontweight='bold')
            axs_wave[2].set_title(f'Forma d\'onda Pulse Rate Filtrata (0.5-4 Hz)')
            axs_wave[2].set_xlabel('Timestamp')
            axs_wave[2].grid(True, alpha=0.3)
            axs_wave[2].fill_between(timestamps_clean, 0, waveform_norm, alpha=0.3, color='#e74c3c')
            
            # Formatta assi temporali
            for ax in axs_wave:
                _format_time_axis(ax)
            
            plt.tight_layout()
            
            # Salva il grafico della forma d'onda
            device_clean = device_name.replace('/', '_').replace(' ', '_')
            filepath = Path(output_dir) / f"PulseWaveform_{device_clean}.png"
            _save_figure(fig_wave, str(filepath))
            plt.close(fig_wave)
            
            print(f"      ✅ Forma d'onda estratta: {len(waveform)} campioni")
            
        except Exception as e:
            print(f"      ⚠️  Errore durante il filtraggio: {e}")

# =====================================================================
# MAPPA INTERATTIVA CON LAYER MULTIPLI
# =====================================================================

print("\n🗺️  Generazione mappa interattiva...")

# Raccogli tutte le coordinate GPS
all_gps = []
for device_type, devices in data_by_device.items():
    for device_name, device_data in devices.items():
        all_gps.extend(device_data['gps_coords'])

if all_gps:
    # Centro mappa
    lat_center = sum(lat for lat, _ in all_gps) / len(all_gps)
    lon_center = sum(lon for _, lon in all_gps) / len(all_gps)
    
    m = folium.Map(location=[lat_center, lon_center], zoom_start=15, 
                   control_scale=True, tiles="OpenStreetMap")
    
    # --- Layer EM: Gradiente CO2 ---
    if 'EM' in data_by_device:
        for device_name, device_data in data_by_device['EM'].items():
            if 'CO2' not in device_data['series']:
                continue
            
            display_name = _get_device_display_name(device_name)
            grp = folium.FeatureGroup(name=f"EM - CO2 ({display_name})").add_to(m)
            
            co2_vals = np.array(device_data['series']['CO2'], dtype=float)
            gps_coords = device_data['gps_coords']
            
            # Limiti colore
            finite = co2_vals[np.isfinite(co2_vals)]
            if finite.size >= 2:
                vmin = float(np.nanpercentile(finite, 5))
                vmax = float(np.nanpercentile(finite, 95))
            else:
                vmin = float(np.nanmin(finite)) if finite.size else 0.0
                vmax = float(np.nanmax(finite)) if finite.size else 1.0
            if vmin == vmax:
                vmax = vmin + 1.0
            
            # Colormap rosso-blu
            colormap = branca.colormap.LinearColormap(
                colors=["#0033FF", "#00B4FF", "#00FFD0", "#7DFF00", "#FFEF00", "#FFA000", "#FF4E00", "#E50000"],
                vmin=vmin, vmax=vmax
            )
            unit_lbl = (units_seen.get('CO2', '') or '').strip() or 'ppm'
            colormap.caption = f"CO2 ({unit_lbl}) - {display_name}"
            colormap.add_to(m)
            
            # Segmenti colorati
            for i in range(1, len(gps_coords)):
                v0, v1 = co2_vals[i-1], co2_vals[i]
                if not (np.isfinite(v0) and np.isfinite(v1)):
                    continue
                v_mid = (v0 + v1) / 2.0
                color = colormap(v_mid)
                folium.PolyLine(
                    [gps_coords[i-1], gps_coords[i]],
                    color=color, weight=7, opacity=0.8,
                    tooltip=f"CO2: {v_mid:.1f} {unit_lbl}"
                ).add_to(grp)
    
    # --- Layer RM: Gradiente Magnitudine Movimento ---
    if 'RM' in data_by_device:
        for device_name, device_data in data_by_device['RM'].items():
            if 'Accel_Magnitude' not in device_data['series']:
                continue
            
            display_name = _get_device_display_name(device_name)
            grp = folium.FeatureGroup(name=f"RM - Movement ({display_name})").add_to(m)
            
            mag_vals = np.array(device_data['series']['Accel_Magnitude'], dtype=float)
            gps_coords = device_data['gps_coords']
            
            # Limiti colore
            finite = mag_vals[np.isfinite(mag_vals)]
            if finite.size >= 2:
                vmin = float(np.nanpercentile(finite, 5))
                vmax = float(np.nanpercentile(finite, 95))
            else:
                vmin = float(np.nanmin(finite)) if finite.size else 0.0
                vmax = float(np.nanmax(finite)) if finite.size else 1.0
            if vmin == vmax:
                vmax = vmin + 1.0
            
            # Colormap verde-giallo
            colormap = branca.colormap.LinearColormap(
                colors=["#00FF00", "#7FFF00", "#FFFF00", "#FFD700", "#FFA500"],
                vmin=vmin, vmax=vmax
            )
            colormap.caption = f"Movement Magnitude (g) - {display_name}"
            colormap.add_to(m)
            
            # Segmenti colorati
            for i in range(1, len(gps_coords)):
                v0, v1 = mag_vals[i-1], mag_vals[i]
                if not (np.isfinite(v0) and np.isfinite(v1)):
                    continue
                v_mid = (v0 + v1) / 2.0
                color = colormap(v_mid)
                folium.PolyLine(
                    [gps_coords[i-1], gps_coords[i]],
                    color=color, weight=7, opacity=0.8,
                    tooltip=f"Movement: {v_mid:.2f} g"
                ).add_to(grp)
    
    # --- Layer SM: Gradiente Heart Rate ---
    if 'SM' in data_by_device:
        for device_name, device_data in data_by_device['SM'].items():
            if 'HR' not in device_data['series']:
                continue
            
            display_name = _get_device_display_name(device_name)
            grp = folium.FeatureGroup(name=f"SM - Heart Rate ({display_name})").add_to(m)
            
            hr_vals = np.array(device_data['series']['HR'], dtype=float)
            gps_coords = device_data['gps_coords']
            
            # Limiti colore
            finite = hr_vals[np.isfinite(hr_vals)]
            if finite.size >= 2:
                vmin = float(np.nanpercentile(finite, 5))
                vmax = float(np.nanpercentile(finite, 95))
            else:
                vmin = float(np.nanmin(finite)) if finite.size else 0.0
                vmax = float(np.nanmax(finite)) if finite.size else 1.0
            if vmin == vmax:
                vmax = vmin + 1.0
            
            # Colormap viola-arancione
            colormap = branca.colormap.LinearColormap(
                colors=["#9B59B6", "#E74C3C", "#E67E22", "#F39C12", "#F1C40F"],
                vmin=vmin, vmax=vmax
            )
            colormap.caption = f"Heart Rate (bpm) - {display_name}"
            colormap.add_to(m)
            
            # Segmenti colorati
            for i in range(1, len(gps_coords)):
                v0, v1 = hr_vals[i-1], hr_vals[i]
                if not (np.isfinite(v0) and np.isfinite(v1)):
                    continue
                v_mid = (v0 + v1) / 2.0
                color = colormap(v_mid)
                folium.PolyLine(
                    [gps_coords[i-1], gps_coords[i]],
                    color=color, weight=7, opacity=0.8,
                    tooltip=f"HR: {v_mid:.0f} bpm"
                ).add_to(grp)
    
    # Controllo layer
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Salva nella directory di output
    map_filepath = Path(output_dir) / 'interactive_devices_map.html'
    m.save(str(map_filepath))
    print(f"✅ Mappa salvata: {map_filepath}")
    webbrowser.open(str(map_filepath), new=2)
else:
    print("⚠️  Nessuna coordinata GPS trovata.")

# Mostra tutti i grafici
print(f"\n✅ Visualizzazione completata!")
print(f"📊 File processati: {len(csv_files)}")
print(f"📈 Grafici generati per {len(data_by_device)} tipi di dispositivi")
if 'EM' in data_by_device:
    print(f"   - Environmental Monitor: {len(data_by_device['EM'])} dispositivi")
if 'RM' in data_by_device:
    print(f"   - Respiratory Monitor: {len(data_by_device['RM'])} dispositivi")
if 'SM' in data_by_device:
    print(f"   - Stress Monitor: {len(data_by_device['SM'])} dispositivi")

print(f"\n✅ Tutti i grafici e mappe sono stati salvati in: {output_dir}")

# Lista tutti i file generati nella directory di output
output_path = Path(output_dir)
png_files = list(output_path.glob("*.png"))
html_files = list(output_path.glob("*.html"))

if png_files or html_files:
    print("\n� File generati:")
    
    # Raggruppa per tipo di dispositivo
    em_files = [f for f in png_files if f.name.startswith('EM_')]
    rm_files = [f for f in png_files if f.name.startswith('RM_')]
    sm_files = [f for f in png_files if f.name.startswith('SM_')]
    pulse_files = [f for f in png_files if f.name.startswith('PulseWaveform_')]
    
    if em_files:
        print(f"   🌍 Environmental Monitor ({len(em_files)} dispositivi):")
        for f in sorted(em_files):
            print(f"      - {f.name}")
    
    if rm_files:
        print(f"   🫁 Respiratory Monitor ({len(rm_files)} dispositivi):")
        for f in sorted(rm_files):
            print(f"      - {f.name}")
    
    if sm_files:
        print(f"   ❤️  Stress Monitor ({len(sm_files)} dispositivi):")
        for f in sorted(sm_files):
            print(f"      - {f.name}")
    
    if pulse_files:
        print(f"   � Pulse Rate Waveforms ({len(pulse_files)} dispositivi):")
        for f in sorted(pulse_files):
            print(f"      - {f.name}")
    
    if html_files:
        print(f"   🗺️  Mappe interattive:")
        for f in sorted(html_files):
            print(f"      - {f.name}")

print(f"\n🚀 Apri la cartella: {output_dir}")
print("💡 Per visualizzare i grafici, apri i file PNG")
print("🌐 Per la mappa interattiva, apri il file HTML")

plt.close('all')  # Chiude tutte le figure per liberare memoria
