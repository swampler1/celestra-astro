from __future__ import annotations

import pandas as pd


# Loading source-parameter table info
#      - Dist_DR3           -> Distance_pc
#      - Mstar_PPVII        -> StellarMass_Msun
#      - vsys_v2            -> SystemicVelocity_kms
#      - PA_deg            -> PA_deg
#      - Inc_deg           -> Inclination_deg
def load_source_parameters_table(csv_path: str) -> dict[str, dict[str, float | str | None]]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Could not load Source-parameter basic-facts table from {csv_path}: {e}")
        return {}

    required_cols = ["Source", "Dist_DR3", "Mstar_PPVII", "vsys_v2", "PA_deg", "Inc_deg"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[WARN] source-parameter table is missing required column '{col}'.")
            return {}

    table: dict[str, dict[str, float | str | None]] = {}

    for _, row in df.iterrows():
        source_raw = row["Source"]
        if pd.isna(source_raw):
            continue
        key = str(source_raw).strip()
        if not key:
            continue

        # Use lowercased key for robust lookup
        key_lower = key.lower()

        table[key_lower] = {
            "Source": key,
            "Distance_pc": None if pd.isna(row["Dist_DR3"]) else float(row["Dist_DR3"]),
            "StellarMass_Msun": None if pd.isna(row["Mstar_PPVII"]) else float(row["Mstar_PPVII"]),
            "SystemicVelocity_kms": None if pd.isna(row["vsys_v2"]) else float(row["vsys_v2"]),
            "PA_deg": None if pd.isna(row["PA_deg"]) else float(row["PA_deg"]),
            "Inclination_deg": None if pd.isna(row["Inc_deg"]) else float(row["Inc_deg"]),
        }

    print(f"[INFO] Loaded Source-parameter basic-facts table with {len(table)} entries.")
    return table


# Looks up source parameters for a target using its exact Source name
def get_source_parameters_for_source_name(
    source_name: str,
    source_parameters: dict[str, dict[str, float | str | None]],
) -> dict[str, float | str | None]:
    if not source_name:
        return {}
    key_lower = source_name.strip().lower()
    return source_parameters.get(key_lower, {})


# Formats source parameters for a target into a plain-text table block
def format_source_parameters_block(
    source_name: str,
    source_parameters: dict[str, dict[str, float | str | None]],
) -> str:
    row = get_source_parameters_for_source_name(source_name, source_parameters)
    if not row:
        return f"No source parameters found for {source_name} in the source-parameter table."

    def fmt_val(val: float | str | None, fmt: str = "{:.2f}") -> str:
        if val is None:
            return "unknown"
        if isinstance(val, (float, int)):
            try:
                return fmt.format(val)
            except Exception:
                return str(val)
        return str(val)

    lines = []
    lines.append(f"Source (catalog)          : {row['Source']}")
    lines.append(f"Distance (pc, DR3)     : {fmt_val(row['Distance_pc'])}")
    lines.append(f"Stellar mass (M_sun)   : {fmt_val(row['StellarMass_Msun'])}")
    lines.append(f"Systemic velocity (km/s): {fmt_val(row['SystemicVelocity_kms'])}")
    lines.append(f"Disk PA (deg)          : {fmt_val(row['PA_deg'])}")
    lines.append(f"Disk inclination (deg) : {fmt_val(row['Inclination_deg'])}")

    return "\n".join(lines)
