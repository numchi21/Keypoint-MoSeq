from jax import config as jax_config
jax_config.update("jax_enable_x64", False)

import numpy as np
import h5py
from pathlib import Path
import keypoint_moseq as kpms


# -----------------------------
# SE AJUSTA ESTO
# -----------------------------
FPS = 30
DATA_ROOT = Path("data/h5")
PROJECT_DIR = Path("kpms_project")


# -----------------------------
# SLEAP LOADER
# tracks: (R, 2, K, T)  -> (T, K, 2, R)
# occ:    (T, R)
# -----------------------------
def load_sleap_h5(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        tracks = np.array(f["/tracks"], dtype=np.float32)            # (R,2,K,T)
        occ = np.array(f["/track_occupancy"], dtype=np.uint8)        # (T,R)
        node_names = f["/node_names"][...]                           # (K,)

    # decode node_names
    bodyparts = []
    for x in np.array(node_names):
        if isinstance(x, (bytes, np.bytes_)):
            bodyparts.append(x.decode("utf-8"))
        else:
            try:
                bodyparts.append(bytes(x).decode("utf-8"))
            except Exception:
                bodyparts.append(str(x))

    # sanity
    if tracks.ndim != 4:
        raise ValueError(f"{h5_path}: tracks.ndim={tracks.ndim}, esperado 4")
    R, D, K, T = tracks.shape
    if D != 2:
        raise ValueError(f"{h5_path}: D={D}, esperado 2 (x,y)")
    if len(bodyparts) != K:
        raise ValueError(f"{h5_path}: len(node_names)={len(bodyparts)} != K={K}")
    if occ.shape != (T, R):
        raise ValueError(f"{h5_path}: occupancy shape {occ.shape} != (T,R)=({T},{R})")

    # reorder to (T,K,2,R)
    tracks = np.transpose(tracks, (3, 2, 1, 0))  # (T,K,2,R)

    # apply occupancy: si occ[t,r]==0 => NaN para todos los keypoints ese frame
    for r in range(R):
        missing = (occ[:, r] == 0)  # (T,)
        if np.any(missing):
            tracks[missing, :, :, r] = np.nan

    # confidences: 1 si válido, 0 si NaN
    conf = np.ones((T, K, R), dtype=np.float32)
    conf[np.isnan(tracks[:, :, 0, :]) | np.isnan(tracks[:, :, 1, :])] = 0.0

    return tracks, conf, bodyparts


def collect_h5_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.h5") if p.is_file()])


def main():
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    h5_files = collect_h5_files(DATA_ROOT)
    if not h5_files:
        raise FileNotFoundError(f"No encontré .h5 en {DATA_ROOT.resolve()}")

    # --- cargar todo y separar por animal: cada track = recording independiente ---
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
                f"Bodyparts distintos entre archivos.\n"
                f"Primero: {bodyparts_ref}\n"
                f"Ahora:   {bodyparts}\n"
                f"Archivo: {h5_path}"
            )

        # split por animal
        for r in range(R):
            rec = f"{h5_path.stem}__animal{r + 1}"
            coordinates_dict[rec] = coords[:, :, :, r]  # (T,K,2)
            confidences_dict[rec] = conf[:, :, r]  # (T,K)
            recording_names.append(rec)

    print("\n=== DEBUG SHAPES ===")
    print("n recordings:", len(recording_names))

    for i, rec in enumerate(recording_names[:5]):
        print(f"rec {i} ({rec}): coords = {coordinates_dict[rec].shape}, confs = {confidences_dict[rec].shape}")

    # checks duros
    assert len(coordinates_dict) == len(confidences_dict) == len(recording_names)
    K0 = coordinates_dict[recording_names[0]].shape[1]

    for rec in recording_names:
        c = coordinates_dict[rec]
        q = confidences_dict[rec]
        assert c.ndim == 3 and c.shape[2] == 2
        assert q.ndim == 2
        assert c.shape[1] == K0 and q.shape[1] == K0

    print("DEBUG SHAPES: OK\n")

    print("DEBUG SHAPES: OK\n")

    # bodyparts definitivos (vienen del h5, pero los fijamos explícitos)
    bodyparts = bodyparts_ref

    # skeleton explícito (nombres reales, no placeholders)
    skeleton = [
        ["nose", "upper_head"],
        ["upper_head", "base_head"],
        ["base_head", "upper_body"],
        ["upper_body", "base_body"],
        ["base_body", "base_tail"],
        ["base_head", "L_ear"],
        ["base_head", "R_ear"],
        ["base_body", "L_hip"],
        ["base_body", "R_hip"],
        ["upper_body", "L_sh"],
        ["upper_body", "R_sh"],
    ]

    anterior_bodyparts = ["nose"]
    posterior_bodyparts = ["base_tail"]
    use_bodyparts = bodyparts  # si luego quiero excluir algo, aquí

    # setup project
    if not PROJECT_DIR.exists() or not (PROJECT_DIR / "config.yml").exists():
        kpms.setup_project(
            str(PROJECT_DIR),
            overwrite=True
        )

    kpms.update_config(
        str(PROJECT_DIR),
        fps=FPS,
        bodyparts=bodyparts,
        use_bodyparts=use_bodyparts,
        skeleton=skeleton,
        anterior_bodyparts=anterior_bodyparts,
        posterior_bodyparts=posterior_bodyparts,
        outlier_scale_factor=6.0,
    )
    config = kpms.load_config(str(PROJECT_DIR))
    cfg = config
    # outlier removal + format
    coordinates, confidences = kpms.outlier_removal(
        coordinates_dict,
        confidences_dict,
        str(PROJECT_DIR),
        overwrite=False,
        **cfg
    )
    data, metadata = kpms.format_data(coordinates, confidences, **cfg)
    from jax_moseq.utils.debugging import convert_data_precision
    data = convert_data_precision(data)

    # sigmasq_loc (centroid)
    kpms.update_config(
        str(PROJECT_DIR),
        sigmasq_loc=kpms.estimate_sigmasq_loc(
            data["Y"], data["mask"], filter_size=cfg["fps"]
        )
    )
    config = kpms.load_config(str(PROJECT_DIR))

    # init + fit
    pca = kpms.fit_pca(data["Y"], data["mask"], **cfg)
    model = kpms.init_model(data, pca=pca, **cfg)

    num_ar_iters = 50
    model, model_name = kpms.fit_model(
        model, data, metadata, str(PROJECT_DIR),
        ar_only=True, num_iters=num_ar_iters
    )

    # full model
    model, data, metadata, current_iter = kpms.load_checkpoint(
        str(PROJECT_DIR), model_name, iteration=num_ar_iters
    )
    model = kpms.update_hypparams(model, kappa=1e4)

    model = kpms.fit_model(
        model, data, metadata, str(PROJECT_DIR),
        model_name=model_name,
        ar_only=False,
        start_iter=current_iter,
        num_iters=current_iter + 500,
    )[0]

    # reindex + results
    kpms.reindex_syllables_in_checkpoint(str(PROJECT_DIR), model_name)
    model, data, metadata, _ = kpms.load_checkpoint(str(PROJECT_DIR), model_name)
    results = kpms.extract_results(model, metadata, str(PROJECT_DIR), model_name)
    kpms.save_results_as_csv(results, str(PROJECT_DIR), model_name)

    # viz (requiere ffmpeg para movies)
    results = kpms.load_results(str(PROJECT_DIR), model_name)
    kpms.generate_trajectory_plots(coordinates, results, str(PROJECT_DIR), model_name, **cfg)
    kpms.generate_grid_movies(results, str(PROJECT_DIR), model_name, coordinates=coordinates, **cfg)
    kpms.plot_similarity_dendrogram(coordinates, results, str(PROJECT_DIR), model_name, **cfg)

    print("DONE:", (PROJECT_DIR / model_name).resolve())


if __name__ == "__main__":
    main()
