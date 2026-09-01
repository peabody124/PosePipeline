"""SAM3 video predictor wrapper for person tracking."""

import os
import tempfile
from pathlib import Path

import numpy as np


def sam3_bounding_boxes(file_path: str, chunk_size: int = 1300, prompt: str = "person") -> list[list[dict]]:
    """Run SAM3 person tracking and return per-frame bounding boxes.

    Processes video in chunks to bound GPU memory on long videos. Track IDs
    are consistent across chunk boundaries via IoU matching on adjacent frames.

    Returns standard tracking format: list[frames] of list[dict] with
    keys track_id, tlbr, tlhw, confidence.
    """
    import cv2
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    predictor = build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            return _run_chunked(predictor, file_path, num_frames, fps, width, height, chunk_size, tmpdir, prompt)
    finally:
        predictor.shutdown()


def _run_chunked(
    predictor,
    file_path: str,
    num_frames: int,
    fps: float,
    width: int,
    height: int,
    chunk_size: int,
    tmpdir: str,
    prompt: str = "person",
) -> list[list[dict]]:
    """Process video in memory-bounded chunks with consistent track IDs across boundaries."""
    import cv2

    all_tracks: list[list[dict]] = [[] for _ in range(num_frames)]
    prev_last_tracks: list[dict] = []
    next_id = 1

    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    chunk_path = os.path.join(tmpdir, "chunk.mp4")
    chunk_start = 0

    while chunk_start < num_frames:
        chunk_end = min(chunk_start + chunk_size, num_frames)
        actual_size = chunk_end - chunk_start

        writer = cv2.VideoWriter(chunk_path, fourcc, fps, (width, height))
        for _ in range(actual_size):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()

        raw_outputs = _run_video_session(predictor, chunk_path, prompt)
        chunk_tracks = _build_tracks(raw_outputs, actual_size)

        id_map, next_id = _match_and_allocate_ids(prev_last_tracks, chunk_tracks[0] if chunk_tracks else [], next_id)
        chunk_tracks, next_id = _remap_track_ids(chunk_tracks, id_map, next_id)

        for i, frame_tracks in enumerate(chunk_tracks):
            all_tracks[chunk_start + i] = frame_tracks

        prev_last_tracks = chunk_tracks[-1] if chunk_tracks else []
        chunk_start = chunk_end

    cap.release()
    return all_tracks


def _match_and_allocate_ids(
    prev_last_tracks: list[dict],
    curr_first_tracks: list[dict],
    next_id: int,
) -> tuple[dict[int, int], int]:
    """Match current chunk's first-frame IDs to previous chunk's last-frame IDs via IoU.

    Returns id_map {curr_id: assigned_id} and the updated next_id counter.
    For the first chunk (prev empty), preserves SAM3's own IDs and sets next_id accordingly.
    """
    if not prev_last_tracks:
        id_map = {d["track_id"]: d["track_id"] for d in curr_first_tracks}
        if curr_first_tracks:
            next_id = max(next_id, max(d["track_id"] for d in curr_first_tracks) + 1)
        return id_map, next_id

    id_map: dict[int, int] = {}
    used_prev_ids: set[int] = set()

    for curr_det in curr_first_tracks:
        curr_id = curr_det["track_id"]
        best_iou = 0.0
        best_prev_id = None

        for prev_det in prev_last_tracks:
            prev_id = prev_det["track_id"]
            if prev_id in used_prev_ids:
                continue
            iou = _bbox_iou(curr_det["tlbr"], prev_det["tlbr"])
            if iou > best_iou:
                best_iou = iou
                best_prev_id = prev_id

        if best_iou > 0.3 and best_prev_id is not None:
            id_map[curr_id] = best_prev_id
            used_prev_ids.add(best_prev_id)
        else:
            id_map[curr_id] = next_id
            next_id += 1

    return id_map, next_id


def _remap_track_ids(
    tracks: list[list[dict]], id_map: dict[int, int], next_id: int
) -> tuple[list[list[dict]], int]:
    """Apply ID remapping across all frames; IDs not in id_map get fresh sequential IDs."""
    live_map = dict(id_map)
    remapped: list[list[dict]] = []

    for frame_tracks in tracks:
        remapped_frame = []
        for det in frame_tracks:
            old_id = det["track_id"]
            if old_id not in live_map:
                live_map[old_id] = next_id
                next_id += 1
            remapped_frame.append({**det, "track_id": live_map[old_id]})
        remapped.append(remapped_frame)

    return remapped, next_id


def _bbox_iou(a_tlbr: np.ndarray, b_tlbr: np.ndarray) -> float:
    """Compute intersection-over-union between two tlbr bounding boxes."""
    x1 = max(float(a_tlbr[0]), float(b_tlbr[0]))
    y1 = max(float(a_tlbr[1]), float(b_tlbr[1]))
    x2 = min(float(a_tlbr[2]), float(b_tlbr[2]))
    y2 = min(float(a_tlbr[3]), float(b_tlbr[3]))

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (float(a_tlbr[2]) - float(a_tlbr[0])) * (float(a_tlbr[3]) - float(a_tlbr[1]))
    area_b = (float(b_tlbr[2]) - float(b_tlbr[0])) * (float(b_tlbr[3]) - float(b_tlbr[1]))
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _run_video_session(predictor, video_path: str, prompt: str, frame_index: int = 0) -> dict:
    """Run SAM3 on a video file with a text prompt; return raw per-frame outputs."""
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=str(Path(video_path)))
    )
    session_id = response["session_id"]
    predictor.handle_request(request=dict(type="reset_session", session_id=session_id))
    predictor.handle_request(
        request=dict(type="add_prompt", session_id=session_id, frame_index=frame_index, text=prompt)
    )
    outputs_per_frame = {}
    for frame_response in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs_per_frame[frame_response["frame_index"]] = frame_response["outputs"]
    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    return outputs_per_frame


def _build_tracks(raw_outputs: dict, num_frames: int) -> list[list[dict]]:
    """Convert raw SAM3 mask outputs to bounding boxes in standard tracking format."""
    tracks: list[list[dict]] = [[] for _ in range(num_frames)]
    for frame_idx, frame_out in raw_outputs.items():
        obj_ids: list[int] = frame_out["out_obj_ids"].tolist()
        raw_masks = frame_out["out_binary_masks"]

        frame_tracks = []
        for i, obj_id in enumerate(obj_ids):
            mask_np = raw_masks[i]
            if hasattr(mask_np, "cpu"):
                mask_np = mask_np.cpu().numpy()
            bool_mask = mask_np.astype(bool)
            coords = np.argwhere(bool_mask)  # (K, 2) of (row, col) = (y, x)
            if len(coords) == 0:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            frame_tracks.append(
                {
                    "track_id": obj_id,
                    "tlbr": np.array([x_min, y_min, x_max, y_max], dtype=float),
                    "tlhw": np.array([x_min, y_min, x_max - x_min, y_max - y_min], dtype=float),
                    "confidence": 1.0,
                }
            )
        if frame_idx < num_frames:
            tracks[frame_idx] = frame_tracks
    return tracks
