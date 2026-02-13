from pathlib import Path
import re
import numpy as np
import h5py
import pandas as pd
import keypoint_moseq as kpms


# =============================
# CARGA EL MODELO YA ENTRENADO
# =============================
# AJUSTA AQUÍ
TRAIN_PROJECT_DIR = Path("kpms_project")        # donde entrenaste
MODEL_NAME = "TU_MODELO_AQUI"                  # nombre del modelo, es el nombre con timestamp (usar el más reciente)
NEW_DATA_ROOT = Path("data_new/h5")            # .h5 nuevos

OUT_DIR = Path("inference_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

APPLY_NUM_ITERS = 300  # inferencia de z(t) en data nueva


# MACHO: Track 2 en estas sesiones, Track 1 en el resto
# (MISMA LISTA que en entrenamiento)

MALE_TRACK2_IDS = {
    "50_S2", "50_S3", "51_S2", "55_S3",
    "85_S3", "86_S2", "92_S2", "92_S3", "128_S3"
}

def extract_session_id_strict(stem: str) -> str:
    m = re.fullmatch(r"(\d+_S[23])", stem)
    if m is None:
        raise ValueError(
            f"\n❌ Nombre de archivo inválido: '{stem}'\n"
            "Se esperaba formato exacto: <numero>_S2 o <numero>_S3"
        )
    return m.group(1)

def male_track_index_from_stem_strict(stem: str) -> tuple[int, str]:
    sid = extract_session_id_strict(stem)
    return (1, sid) if (sid in MALE_TRACK2_IDS) else (0, sid)   # 1=Track2, 0=Track1


# =============================
# CARGA NUEVOS .H5
# =============================
# SLEAP LOADER: (R,2,K,T) -> (T,K,2,R) + occupancy -> NaNs
def load_sleap_h5(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        tracks = np.array(f["/tracks"], dtype=np.float32)            # (R,2,K,T)
        occ = np.array(f["/track_occupancy"], dtype=np.uint8)        # (T,R)
        node_names = f["/node_names"][...]                           # (K,)

    bodyparts = []
    for x in np.array(node_names):
        if isinstance(x, (bytes, np.bytes_)):
            bodyparts.append(x.decode("utf-8"))
        else:
            try:
                bodyparts.append(bytes(x).decode("utf-8"))
            except Exception:
                bodyparts.append(str(x))

    if tracks.ndim != 4:
        raise ValueError(f"{h5_path}: tracks.ndim={tracks.ndim}, esperado 4")
    R, D, K, T = tracks.shape
    if D != 2:
        raise ValueError(f"{h5_path}: D={D}, esperado 2 (x,y)")
    if R != 2:
        raise ValueError(f"{h5_path}: R={R}, esperado 2 tracks (animal1/animal2)")
    if len(bodyparts) != K:
        raise ValueError(f"{h5_path}: len(node_names)={len(bodyparts)} != K={K}")
    if occ.shape != (T, R):
        raise ValueError(f"{h5_path}: occupancy shape {occ.shape} != (T,R)=({T},{R})")

    tracks = np.transpose(tracks, (3, 2, 1, 0))  # (T,K,2,R)

    for r in range(R):
        missing = (occ[:, r] == 0)
        if np.any(missing):
            tracks[missing, :, :, r] = np.nan

    conf = np.ones((T, K, R), dtype=np.float32)
    conf[np.isnan(tracks[:, :, 0, :]) | np.isnan(tracks[:, :, 1, :])] = 0.0

    return tracks, conf, bodyparts


def collect_h5_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.h5") if p.is_file()])


# =============================
# STATE HISTORY (run-length)
# =============================
def state_history_from_z(z: np.ndarray):
    z = np.asarray(z)
    if z.ndim != 1:
        raise ValueError(f"z debe ser 1D, got {z.shape}")
    T = len(z)
    if T == 0:
        return []

    change = np.r_[True, z[1:] != z[:-1]]
    starts = np.where(change)[0]
    ends = np.r_[starts[1:] - 1, T - 1]

    return [
        {
            "state": int(z[s]),
            "start_frame": int(s),
            "end_frame": int(e),
            "length_frames": int(e - s + 1),
        }
        for s, e in zip(starts, ends)
    ]


def main():
    # 1) cargar config + modelo entrenado
    cfg = kpms.load_config(str(TRAIN_PROJECT_DIR))
    model, _, _, _ = kpms.load_checkpoint(str(TRAIN_PROJECT_DIR), MODEL_NAME)

    # 2) cargar .h5 nuevos, quedarte SOLO con macho
    h5_files = collect_h5_files(NEW_DATA_ROOT)
    if not h5_files:
        raise FileNotFoundError(f"No encontré .h5 en {NEW_DATA_ROOT.resolve()}")

    coordinates_dict = {}
    confidences_dict = {}
    recording_names = []
    bodyparts_ref = None

    for h5_path in h5_files:
        coords, conf, bodyparts = load_sleap_h5(h5_path)  # coords: (T,K,2,R)
        T, K, _, R = coords.shape

        if bodyparts_ref is None:
            bodyparts_ref = bodyparts
        elif bodyparts != bodyparts_ref:
            raise ValueError(
                f"Bodyparts distintos entre archivos nuevos.\n"
                f"Primero: {bodyparts_ref}\n"
                f"Ahora:   {bodyparts}\n"
                f"Archivo: {h5_path}"
            )

        male_r, sid = male_track_index_from_stem_strict(h5_path.stem)
        if not (0 <= male_r < R):
            raise ValueError(f"{h5_path.name}: male_r={male_r} fuera de rango para R={R}")

        rec = f"{sid}__male_track{male_r + 1}"
        coordinates_dict[rec] = coords[:, :, :, male_r]      # (T,K,2)
        confidences_dict[rec] = conf[:, :, male_r]           # (T,K)
        recording_names.append(rec)

        print(f"[MALE ONLY] {h5_path.name} -> {rec}  T={T}  K={K}")

    # 3) mismo prepro + format data
    coordinates, confidences = kpms.outlier_removal(
        coordinates_dict,
        confidences_dict,
        str(TRAIN_PROJECT_DIR),
        overwrite=False,
        **cfg
    )
    data_new, metadata_new = kpms.format_data(coordinates, confidences, **cfg)

    # 4) obtener syllable/state sequence por data set
    results_new, _ = kpms.apply_model(
        model,
        data_new,
        metadata_new,
        project_dir=str(TRAIN_PROJECT_DIR),
        model_name=MODEL_NAME,
        num_iters=APPLY_NUM_ITERS,
        save_results=True,
        results_path=str(OUT_DIR / "results_newdata.h5"),
        verbose=True,
        return_model=True,
        **cfg
    )

    # 5) guardar z(t) y state history por recording
    histories_dir = OUT_DIR / "state_histories"
    histories_dir.mkdir(parents=True, exist_ok=True)

    for rec in recording_names:
        z = np.asarray(results_new[rec]["syllable"])  # (T,)
        df_frames = pd.DataFrame({"frame": np.arange(len(z), dtype=int), "state": z.astype(int)})
        df_frames.to_csv(histories_dir / f"{rec}__z_frames.csv", index=False)

        hist = state_history_from_z(z)
        pd.DataFrame(hist).to_csv(histories_dir / f"{rec}__state_history.csv", index=False)

        print(f"[SAVED] {rec}: frames={len(z)} bouts={len(hist)}")

    print("\n DONE")
    print("H5:", (OUT_DIR / "results_newdata.h5").resolve())
    print("CSVs:", histories_dir.resolve())


if __name__ == "__main__":
    main()