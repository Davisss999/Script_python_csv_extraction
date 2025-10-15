# questo file serve a leggere file .csv generati da app Anthem e a plottare mappa con spostamenti persona

import csv
import math
import re
import webbrowser
from datetime import datetime

import branca
import folium
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

csv_path = r"C:\Users\david\Desktop portatile\CSV_Python\EM.csv"  # file corretto

# --- righe da saltare ---------------------------------
# Indici riga 1-based
SKIP_ROWS = set(range(1,101))

# Timestamp esatti da saltare
SKIP_TIMESTAMPS = set()


def should_skip(row_idx: int, row: dict) -> bool:
    return False
# ------------------------------------------------------------------------------

# --- formato timestamp completo per visualizzazione ---
DISPLAY_TS_FMT = '%d/%m/%Y %H:%M:%S'

timestamps_raw = []
gps_coords     = []   # [(lat, lon), …]
device_names   = []
row_metrics    = []   # per riga: {metrica_normalizzata: valore_float}
units_seen     = {}   # metrica_normalizzata -> unità (prima non-vuota vista)

# -------- helpers --------
def parse_ts(ts):
    # accetta ISO con/ senza millisecondi
    t = ts.strip().replace('Z', '')
    try:
        return datetime.fromisoformat(t)
    except ValueError:
        return datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f')

# normalizza trattini/alias
def normalize_metric_name(name: str) -> str:
    h = re.sub(r'[\u2010-\u2015\u2212]', '-', name.strip())   # trattini → '-'
    h = re.sub(r'\s*-\s*', '-', h)
    h = re.sub(r'\s+', ' ', h)
    no_space_up = re.sub(r'\s+', '', h).upper()
    if re.fullmatch(r'VOC-?I(NDEX)?', no_space_up):
        return 'VOC index'
    if re.fullmatch(r'NOX-?I(NDEX)?', no_space_up) or re.fullmatch(r'NOXINDEX', no_space_up):
        return 'NOx index'
    return h

# Regex generale: "Nome: numero [unità facoltativa]" fino alla virgola successiva
PATTERN = re.compile(r'\s*([^:,]+?)\s*:\s*([-+]?\d*\.?\d+)\s*([^,]*)')

# -------- lettura CSV --------
skipped_count = 0
with open(csv_path, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)

    # row_idx: indice 1-based delle righe DATI (header escluso)
    for row_idx, row in enumerate(reader, start=1):
        # --- Salta righe scelte ---
        raw_ts = (row.get('TIMESTAMP') or '').strip()
        if (row_idx in SKIP_ROWS) or (raw_ts in SKIP_TIMESTAMPS) or should_skip(row_idx, row):
            skipped_count += 1
            continue
        # -----------------------------------

        ts = raw_ts
        gps = (row.get('GPS_COORDINATES') or '').strip()
        device = (row.get('DEVICE_NAME') or row.get('DEVICE') or '').strip()
        data_str = (row.get('DATA received') or '').strip()

        if not ts or not gps or not data_str:
            continue

        # parse GPS
        try:
            lat_s, lon_s = [s.strip() for s in gps.split(',', 1)]
            lat, lon = float(lat_s), float(lon_s)
        except Exception:
            continue

        metrics_dict = {}
        for name, num, unit in PATTERN.findall(data_str):
            norm = normalize_metric_name(name)
            try:
                val = float(num)
            except ValueError:
                continue
            metrics_dict[norm] = val
            unit = unit.strip()
            if unit and norm not in units_seen:
                units_seen[norm] = unit

        if not metrics_dict:
            continue

        timestamps_raw.append(ts)
        gps_coords.append((lat, lon))
        device_names.append(device)
        row_metrics.append(metrics_dict)

print(f"Righe saltate: {skipped_count}")

# -------- ordinamento temporale --------
timestamps_dt = [parse_ts(ts) for ts in timestamps_raw]
order = sorted(range(len(timestamps_dt)), key=lambda i: timestamps_dt[i])
timestamps_dt = [timestamps_dt[i] for i in order]
gps_coords     = [gps_coords[i] for i in order]
device_names   = [device_names[i] for i in order]
row_metrics    = [row_metrics[i] for i in order]
timestamps_raw = [timestamps_raw[i] for i in order]

# -------- elenco metriche + serie allineate --------
metric_order = []
seen = set()
for d in row_metrics:
    for k in d.keys():
        if k not in seen:
            seen.add(k)
            metric_order.append(k)

series = {m: [] for m in metric_order}
for d in row_metrics:
    for m in metric_order:
        series[m].append(d.get(m, np.nan))

print("Metriche trovate:", metric_order)
print("\n================= CONTENUTO VETTORI METRICHE =================")
for m in metric_order:
    unit = (units_seen.get(m, '') or '').strip()
    label = f"{m} ({unit})" if unit else m

    vals_list = series[m]
    n_total = len(vals_list)
    n_valid = sum(1 for v in vals_list if not (isinstance(v, float) and math.isnan(v)))

    print(f"\n{label} — n={n_total} (validi: {n_valid})")
    print(vals_list)  # Lista completa; i mancanti appaiono come 'nan'

# -------- helper per asse-tempo: timestamp COMPLETO --------
def _format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    # formatter esplicito completo gg/mm/aaaa hh:mm:ss
    ax.xaxis.set_major_formatter(mdates.DateFormatter(DISPLAY_TS_FMT))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha('right')

# -------- plotting automatico (griglia combinata) --------
n = len(metric_order)
if n == 0:
    raise SystemExit("Nessuna metrica trovata nella colonna 'DATA received'.")

ncols = 2
nrows = math.ceil(n / ncols)
fig, axs = plt.subplots(nrows, ncols, figsize=(14, max(6, nrows * 3)), sharex='all', squeeze=False)

# normalizza axs in matrice 2D
if nrows == 1 and ncols == 1:
    axs = np.array([[axs]])
elif nrows == 1:
    axs = np.array([axs])

for i, m in enumerate(metric_order):
    r, c = divmod(i, ncols)
    ax = axs[r, c]
    ax.plot(timestamps_dt, series[m])
    unit = (units_seen.get(m, '') or '').strip()
    ylabel = f"{m} ({unit})" if unit else m
    ax.set_ylabel(ylabel)
    _format_time_axis(ax)
    ax.grid(True, alpha=0.3)

# nascondi subplot in eccesso
for j in range(n, nrows * ncols):
    r, c = divmod(j, ncols)
    axs[r, c].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.20, top=0.95, left=0.08, right=0.98, hspace=0.35, wspace=0.25)

# -------- plotting singoli (una finestra per metrica, x = timestamp completo) --------
for m in metric_order:
    fig_i, ax_i = plt.subplots(figsize=(12, 4.5))
    ax_i.plot(timestamps_dt, series[m])
    unit = (units_seen.get(m, '') or '').strip()
    ylabel = f"{m} ({unit})" if unit else m
    ax_i.set_title(ylabel)
    ax_i.set_xlabel("Timestamp (gg/mm/aaaa hh:mm:ss)")
    ax_i.set_ylabel(ylabel)
    _format_time_axis(ax_i)
    ax_i.grid(True, alpha=0.3)
    fig_i.tight_layout()

# Mostra tutte le figure (griglia + singoli)
plt.show()

# -------- mappa --------

if gps_coords:
    lat_center = sum(lat for lat, _ in gps_coords) / len(gps_coords)
    lon_center = sum(lon for _, lon in gps_coords) / len(gps_coords)

    m = folium.Map(location=[lat_center, lon_center], zoom_start=13, control_scale=True, tiles="OpenStreetMap")

    # Marker puntuali con popup (timestamp completo + CO2 se disponibile)
    co2_vals = np.array([d.get('CO2', float('nan')) for d in row_metrics], dtype=float)
    SHOW_MARKERS = False  # metti True se vuoi visualizzare markers
    if SHOW_MARKERS:
        for (lat, lon), ts, v in zip(gps_coords, timestamps_raw, co2_vals):
            ts_str = parse_ts(ts).strftime(DISPLAY_TS_FMT)
            co2_txt = f"<br><b>CO2:</b> {v:.1f} {(units_seen.get('CO2','') or '').strip()}" if np.isfinite(v) else ""
            folium.CircleMarker(
                [lat, lon],
                radius=4,
                color="#2a81cb",
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(f"<b>{ts_str}</b>{co2_txt}", max_width=260),
            ).add_to(m)

    # Traccia semplice originale
    folium.PolyLine(gps_coords, color='blue', weight=2, opacity=0.5, tooltip="Percorso").add_to(m)

    # ===== Layer con gradiente CO2 =====
    grp = folium.FeatureGroup(name="CO2 (gradient)").add_to(m)

    # Limiti di colore robusti (5°–95° percentile) con fallback
    finite = co2_vals[np.isfinite(co2_vals)]
    if finite.size >= 2:
        vmin = float(np.nanpercentile(finite, 5))
        vmax = float(np.nanpercentile(finite, 95))
    else:
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0

    # Colormap
    print(vmin,vmax)
    colormap = branca.colormap.LinearColormap(
        colors=["#0033FF", "#00B4FF", "#00FFD0", "#7DFF00", "#FFEF00", "#FFA000", "#FF4E00", "#E50000"],  # blu→rosso
        vmin=vmin, vmax=vmax
    )
    unit_lbl = (units_seen.get('CO2','') or '').strip() or 'arb. units'
    colormap.caption = f"CO2 ({unit_lbl}) — blue=low, red=high"
    colormap.add_to(m)

    # Disegna la polilinea a segmenti colorati per CO2
    for i in range(1, len(gps_coords)):
        v0, v1 = co2_vals[i-1], co2_vals[i]
        if not (np.isfinite(v0) and np.isfinite(v1)):
            continue
        v_mid = (v0 + v1) / 2.0
        color = colormap(v_mid)
        folium.PolyLine(
            [gps_coords[i-1], gps_coords[i]],
            color=color, weight=7, opacity=1,
            tooltip=f"CO2 ~ {v_mid:.1f} {unit_lbl}"
        ).add_to(grp)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save('visited_points_map.html')
    webbrowser.open('visited_points_map.html', new=2)
else:
    print("Nessuna coordinata GPS valida trovata, salto la mappa.")
