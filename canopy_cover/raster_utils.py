"""Raster alignment and CHM computation helpers."""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds


def load_aligned_stack(orthophoto_path, dsm_path, dtm_path, ref="dsm"):
    """Load orthophoto (RGB), DSM, and DTM resampled onto a single common grid.

    The three rasters are not pixel-aligned on disk (differing width/height/
    bounds), so DTM and orthophoto are reprojected/resampled onto the DSM's
    grid (or whichever `ref` in {"dsm", "dtm"} is chosen).

    Returns
    -------
    dict with keys: rgb (3,H,W) uint8, dsm (H,W) float32, dtm (H,W) float32,
    transform, crs, nodata masks (dsm_mask, dtm_mask) as boolean arrays
    (True = nodata/invalid).
    """
    with rasterio.open(dsm_path) as dsm_src, rasterio.open(dtm_path) as dtm_src, \
            rasterio.open(orthophoto_path) as ortho_src:

        ref_src = dsm_src if ref == "dsm" else dtm_src
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_shape = (ref_src.height, ref_src.width)

        dsm = dsm_src.read(1).astype(np.float32)
        dsm_nodata = dsm_src.nodata
        if ref_src is not dsm_src:
            dsm_dst = np.full(ref_shape, dsm_nodata if dsm_nodata is not None else np.nan, dtype=np.float32)
            reproject(
                source=dsm, destination=dsm_dst,
                src_transform=dsm_src.transform, src_crs=dsm_src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                src_nodata=dsm_nodata, dst_nodata=dsm_nodata,
                resampling=Resampling.bilinear,
            )
            dsm = dsm_dst

        dtm = dtm_src.read(1).astype(np.float32)
        dtm_nodata = dtm_src.nodata
        if ref_src is not dtm_src:
            dtm_dst = np.full(ref_shape, dtm_nodata if dtm_nodata is not None else np.nan, dtype=np.float32)
            reproject(
                source=dtm, destination=dtm_dst,
                src_transform=dtm_src.transform, src_crs=dtm_src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                src_nodata=dtm_nodata, dst_nodata=dtm_nodata,
                resampling=Resampling.bilinear,
            )
            dtm = dtm_dst

        rgb = np.zeros((3, ref_shape[0], ref_shape[1]), dtype=np.uint8)
        n_bands = min(3, ortho_src.count)
        ortho_data = ortho_src.read(list(range(1, n_bands + 1)))
        reproject(
            source=ortho_data, destination=rgb[:n_bands],
            src_transform=ortho_src.transform, src_crs=ortho_src.crs,
            dst_transform=ref_transform, dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )

        dsm_mask = (dsm == dsm_nodata) if dsm_nodata is not None else np.isnan(dsm)
        dtm_mask = (dtm == dtm_nodata) if dtm_nodata is not None else np.isnan(dtm)

        return {
            "rgb": rgb,
            "dsm": dsm,
            "dtm": dtm,
            "transform": ref_transform,
            "crs": ref_crs,
            "dsm_mask": dsm_mask,
            "dtm_mask": dtm_mask,
        }


def compute_chm(dsm: np.ndarray, dtm: np.ndarray, dsm_mask: np.ndarray, dtm_mask: np.ndarray):
    """CHM = DSM - DTM. Returns (chm, valid_mask) where valid_mask True = usable pixel."""
    valid = ~(dsm_mask | dtm_mask)
    chm = np.full(dsm.shape, np.nan, dtype=np.float32)
    chm[valid] = dsm[valid] - dtm[valid]
    return chm, valid
