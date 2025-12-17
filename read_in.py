import os
import re
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import TheilSenRegressor
import matplotlib.pyplot as plt

# ---------- CONSTANTS ----------
ROOT_DIR = "regi/Balaton"              # <- start here instead of one BASE_DIR
CALIB_PATH = "szamitas/calib_scores.txt"
target_wavelengths = np.arange(380, 901)
h = 6.626e-34
c = 299792458
mol = 6e23

# ---------- LOAD CALIBRATION ONCE ----------
cal_vals = []
with open(CALIB_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        cal_vals.append(float(line.replace(",", ".")) * np.pi)

cal_vals = np.array(cal_vals)
if len(cal_vals) != len(target_wavelengths):
    raise ValueError(f"Calibration length {len(cal_vals)} != {len(target_wavelengths)} wavelengths")

calibration_dict = dict(zip(target_wavelengths, cal_vals))


def process_spectrum_file(path: str) -> pd.Series:
    """
    Parse one txt file and return a Series:
        index   = wavelength (380..900)
        values  = nmol/m2
    """
    header = {}
    data = []
    data_started = False

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            line = line.replace("\t", " ")
            if not line:
                continue

            if line.startswith(">"):
                data_started = True
                continue

            if not data_started:
                # Header part
                if ":" in line:
                    key, value = line.split(":", 1)
                    header[key.strip()] = value.strip()
                else:
                    header.setdefault("Misc", []).append(line)
            else:
                # Data part
                parts = line.split()
                if len(parts) >= 2:
                    if "End" in parts[0]:
                        break
                    wl = float(parts[0].replace(",", "."))
                    intensity = float(parts[1].replace(",", "."))
                    data.append((wl, intensity))

    df = pd.DataFrame(data, columns=["wavelength", "intensity"])
    df = df.sort_values("wavelength").drop_duplicates(subset="wavelength")

    x = df["wavelength"].to_numpy()
    y = df["intensity"].to_numpy()

    # Interpolate to 380..900
    y_interp = np.interp(target_wavelengths, x, y)
    df_interp = pd.DataFrame({
        "wavelength": target_wavelengths,
        "intensity": y_interp,
    })

    # Integration time from header
    time = np.float32(header["Integration Time (usec)"].split()[0])

    # Calibration values (already matched to wavelengths)
    df_interp["calib_scores"] = cal_vals
    df_interp["calib_data"] = df_interp["intensity"] * df_interp["calib_scores"] / time

    # Convert to energy and nmol/m2
    df_interp["m"] = df_interp["wavelength"] / 1e9  # nm → m
    df_interp["E"] = h * c / df_interp["m"]
    df_interp["counted"] = df_interp["calib_data"] / df_interp["E"]
    df_interp["nmol/m2"] = df_interp["counted"] / mol * 1e6

    # Return just the nmol/m2 as a Series indexed by wavelength
    series = df_interp.set_index("wavelength")["nmol/m2"]
    return series


def fit_attenuation(depths, intensities):
    mask = intensities > 0
    depths = np.array(depths)[mask]
    intensities = np.array(intensities)[mask]

    if len(depths) < 3:
        return np.nan, np.nan, np.nan

    X = depths.reshape(-1, 1)
    y = np.log(intensities)

    model = TheilSenRegressor()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    # compute R2 manually
    y_pred = model.predict(X)
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return slope, intercept, R2


# ====== LOOP OVER ALL LEAF FOLDERS (STATIONS) ======
pattern = re.compile(
    r"_R_(\d)_(\d+)mm(?:\s*hull(?:am|ám)|_felho)?\.txt$",
    re.IGNORECASE
)

for BASE_DIR, subdirs, files in os.walk(ROOT_DIR):
    # pick only dirs that actually contain txt spectra
    txt_files = [f for f in files if f.lower().endswith(".txt")]
    if not txt_files:
        continue

    station_name = os.path.basename(BASE_DIR)
    print(f"\n=== Processing station folder: {BASE_DIR} ===")

    # --- same mechanics as before, just inside loop ---
    rep1_series_by_depth = {}
    rep2_series_by_depth = {}

    for fname in txt_files:
        m = pattern.search(fname)
        if not m:
            continue

        rep = int(m.group(1))          # 1 or 2
        depth_mm = int(m.group(2))     # 0, 250, 500, ...

        full_path = os.path.join(BASE_DIR, fname)
        series = process_spectrum_file(full_path)   # wavelength-indexed nmol/m2

        if rep == 1:
            rep1_series_by_depth[depth_mm] = series
        elif rep == 2:
            rep2_series_by_depth[depth_mm] = series
        
        depths1 = set(rep1_series_by_depth.keys())
        depths2 = set(rep2_series_by_depth.keys())
        common_depths = sorted(depths1 & depths2)

    if not common_depths:
        print(f"  Skipping {BASE_DIR}: no common depths between replicates.")
        continue

    if len(common_depths) < 3:
        print(f"  Warning in {BASE_DIR}: only {len(common_depths)} common depths, results may be unstable.")

    if depths1 != depths2:
        print(f"  Note: depth mismatch in {BASE_DIR}.")
        print(f"    rep1 depths: {sorted(depths1)}")
        print(f"    rep2 depths: {sorted(depths2)}")
        print(f"    using only common depths: {common_depths}")

    if not rep1_series_by_depth or not rep2_series_by_depth:
        print(f"  Skipping {station_name}: missing replicate 1 or 2.")
        continue

    index = target_wavelengths

    df_rep1 = pd.DataFrame(index=index)
    df_rep2 = pd.DataFrame(index=index)

    for depth in common_depths:
        s1 = rep1_series_by_depth[depth]
        s2 = rep2_series_by_depth[depth]
        df_rep1[f"{depth}mm"] = s1.reindex(index)
        df_rep2[f"{depth}mm"] = s2.reindex(index)


    print("  Replicate 1 shape:", df_rep1.shape)
    print("  Replicate 2 shape:", df_rep2.shape)

    depth_cols = df_rep1.columns
    depths_m = np.array([int(col.replace("mm", "")) / 1000 for col in depth_cols])

    results_rep1 = []
    results_rep2 = []

    for wl in df_rep1.index:
        intens1 = df_rep1.loc[wl].values.astype(float)
        intens2 = df_rep2.loc[wl].values.astype(float)

        slope1, intercept1, R2_1 = fit_attenuation(depths_m, intens1)
        slope2, intercept2, R2_2 = fit_attenuation(depths_m, intens2)

        results_rep1.append((wl, slope1, intercept1, R2_1))
        results_rep2.append((wl, slope2, intercept2, R2_2))

    df_fit_rep1 = pd.DataFrame(results_rep1, columns=["wavelength", "slope", "intercept", "R2"])
    df_fit_rep2 = pd.DataFrame(results_rep2, columns=["wavelength", "slope", "intercept", "R2"])

    # Convert slopes → Kd (m^-1)
    df_fit_rep1["Kd"] = -df_fit_rep1["slope"]
    df_fit_rep2["Kd"] = -df_fit_rep2["slope"]

    # Combine into one table
    df_Kd = pd.DataFrame({
        "wavelength": df_fit_rep1["wavelength"].values,
        "Kd_rep1": df_fit_rep1["Kd"].values,
        "Kd_rep2": df_fit_rep2["Kd"].values,
        "Kd_mean": 0.5 * (df_fit_rep1["Kd"].values + df_fit_rep2["Kd"].values),
    })

    # Smooth (mechanics unchanged, just keep it)
    df_Kd["Kd_smooth"] = savgol_filter(df_Kd["Kd_mean"], 31, 3)

    # ---- SAVE CSV + PLOT WITH STATION NAME ----
    csv_name = f"Kd_spectrum_{station_name}.csv"
    fig_name = f"Kd_spectrum_{station_name}.png"

    csv_path = os.path.join(BASE_DIR, csv_name)
    fig_path = os.path.join(BASE_DIR, fig_name)

    df_Kd.to_csv(csv_path, index=False)
    print(f"  Saved Kd spectrum csv to: {csv_path}")

    plt.figure(figsize=(7, 4))
    plt.plot(df_Kd["wavelength"], df_Kd["Kd_mean"], label="Mean Kd", linewidth=1.5)

    plt.axvspan(400, 700, alpha=0.1, label="PAR (400–700 nm)")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel(r"$K_d(\lambda)$ (m$^{-1}$)")
    plt.title("Vertical diffuse attenuation spectrum")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  Saved Kd spectrum plot to: {fig_path}")
