from __future__ import annotations

from contextlib import contextmanager, nullcontext
import cv2
from .math_utils import interpolate_poses
import numpy as np
from pathlib import Path
import subprocess
from typing import List, Optional, Tuple, Union
import tqdm

class VideoGenerator:
    """
    Utility class for generating videos.
    """

    background: np.ndarray
    fps: int
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    save_path: Optional[str]
    dot_radius_px: float
    glow_radius_px: float
    fade_trail_decay: float
    fade_trail_floor: float
    constant_trail_width_px: float
    fade_trail_width_px: float

    def __init__(self, background: np.ndarray, fps: int, xlim: Tuple[float, float], ylim: Tuple[float, float],
                 save_path: Optional[str] = None, dot_radius_px: float = 8.0, glow_radius_px: float = 20.0,
                 fade_trail_decay: float = 0.80, fade_trail_floor: float = 0.0,
                 constant_trail_width_px: float = 2.0, fade_trail_width_px: float = 2.0) -> None:
        """
        Args:
            background: (H, W, 3) BGR background raster, drawn under every frame.
            fps: Output video frame rate.
            xlim: (x_min, x_max), in world units, that background was rendered at.
            ylim: (y_min, y_max), in world units, that background was rendered at.
            save_path: If provided, video is encoded and saved to this path (.mp4).
                Otherwise, generate() plays the video back live in a window.
            dot_radius_px: Radius of the solid marker at each robot's current position.
            glow_radius_px: Radius of the blurred glow halo behind each marker.
            fade_trail_decay: Per-frame multiplicative decay of the fading trail's opacity, in (0, 1).
            fade_trail_floor: Opacity the fading trail decays to and then holds at, in [0, 1).
            constant_trail_width_px: Width of the permanent, never-fading trail.
            fade_trail_width_px: Width of the fading trail.
        """
        self.background = background
        self.fps = fps
        self.xlim = xlim
        self.ylim = ylim
        self.save_path = save_path
        self.dot_radius_px = dot_radius_px
        self.glow_radius_px = glow_radius_px
        self.fade_trail_decay = fade_trail_decay
        self.fade_trail_floor = fade_trail_floor
        self.constant_trail_width_px = constant_trail_width_px
        self.fade_trail_width_px = fade_trail_width_px

    def compute_frame_times(self, time_ranges: List[Tuple[float, float]], video_duration_sec: float) -> np.ndarray:
        """
        Builds a uniform grid of frame timestamps (seconds) spanning the union
        of all given timestamp ranges, sized so that video_duration_sec of
        output at self.fps covers that whole span.

        Args:
            time_ranges: (start, end) timestamp pair for each trajectory.
            video_duration_sec: Target duration of the output video, in seconds.

        Returns:
            Frame timestamps, evenly spaced from the earliest start time to the
            latest end time across all time_ranges.
        """
        if video_duration_sec <= 0:
            raise ValueError("video_duration_sec must be positive")

        overall_start = min(r[0] for r in time_ranges)
        overall_end = max(r[1] for r in time_ranges)
        if overall_end <= overall_start:
            raise ValueError("Combined timestamp range must be positive")

        num_frames = max(int(round(video_duration_sec * self.fps)), 2)
        return np.linspace(overall_start, overall_end, num_frames)

    @staticmethod
    def reencode_to_h264(path: Union[Path, str]) -> None:
        """
        Re-encodes an existing .mp4 file to H.264/AVC (yuv420p), overwriting it
        in place, via the system ffmpeg binary.

        cv2.VideoWriter's "mp4v" fourcc (used elsewhere in this class, and in
        ImageData.to_mp4) produces MPEG-4 Part 2, not H.264 -- OpenCV's
        automatic H.264 encoder selection is unreliable across systems (e.g.
        it may pick an unusable hardware v4l2m2m encoder instead of libx264),
        so this re-encodes after the fact via ffmpeg directly instead. Many
        web/Office video players (e.g. PowerPoint for the web) can't decode
        MPEG-4 Part 2, while H.264 in an MP4 container is far more broadly
        compatible.

        Args:
            path: Path to the existing .mp4 file, overwritten in place with
                the H.264-encoded version.

        Raises:
            RuntimeError: If ffmpeg/ffprobe are not installed, or the re-encode fails.
        """
        path = Path(path)
        temp_path = path.with_suffix('.tmp_h264.mp4')

        # Look up the input's duration via ffprobe, to size the progress bar
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                check=True, capture_output=True, text=True)
            duration_sec = float(probe.stdout.strip())
        except FileNotFoundError:
            raise RuntimeError("ffprobe is required to re-encode to H.264, but was not found on PATH.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffprobe failed to read the duration of {path}:\n{e.stderr}")

        # Run ffmpeg, streaming its machine-readable progress (newline-delimited
        # out_time_ms=... blocks) on stdout to drive a tqdm bar. Popen is used as a
        # context manager so its stdout/stderr pipes are always closed on exit.
        try:
            process_ctx = subprocess.Popen(
                ["ffmpeg", "-y", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-progress", "pipe:1", "-nostats", str(temp_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except FileNotFoundError:
            raise RuntimeError("ffmpeg is required to re-encode to H.264, but was not found on PATH.")

        pbar = tqdm.tqdm(total=round(duration_sec, 2), desc="Re-encoding to H.264...", unit=" sec")
        try:
            with process_ctx as process:
                prev_out_time_sec = 0.0
                for line in process.stdout:
                    key, _, value = line.strip().partition('=')
                    if key == "out_time_ms":
                        out_time_sec = min(int(value) / 1_000_000, duration_sec)
                        pbar.update(out_time_sec - prev_out_time_sec)
                        prev_out_time_sec = out_time_sec
                stderr_output = process.stderr.read()
        finally:
            pbar.close()

        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to re-encode {path} to H.264:\n{stderr_output}")

        temp_path.replace(path)

    @staticmethod
    @contextmanager
    def open_video_writer(output_path: Union[Path, str], fps: float, frame_size: Tuple[int, int]):
        """
        Context manager yielding a cv2.VideoWriter to write frames to, that
        re-encodes to H.264/AVC at output_path on a clean exit and cleans up
        its temporary file -- shared by VideoGenerator.generate() and
        ImageData.to_mp4 so both produce PowerPoint/web-compatible video
        without duplicating the write-then-re-encode logic (see
        reencode_to_h264). Frames are first written, via cv2's "mp4v" fourcc,
        to a temporary raw .mp4 next to output_path, then reencode_to_h264
        converts it in place before it's moved to output_path.

        Args:
            output_path: Final destination for the H.264-encoded .mp4.
            fps: Frame rate to open the writer at.
            frame_size: (width, height) in pixels.

        Yields:
            An opened cv2.VideoWriter to write frames to.

        Raises:
            ValueError: If the underlying video writer can't be opened.
            RuntimeError: If ffmpeg is not installed, or the re-encode fails.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix('.tmp_raw.mp4')

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_path), fourcc, fps, frame_size)
        if not writer.isOpened():
            raise ValueError(f"Could not open a video writer for output_path: {output_path}")

        try:
            yield writer
        except Exception:
            writer.release()
            temp_path.unlink(missing_ok=True)
            raise
        else:
            writer.release()
            VideoGenerator.reencode_to_h264(temp_path)
            temp_path.replace(output_path)

    @staticmethod
    def _interpolate_xy_at_times(timestamps: np.ndarray, xy: np.ndarray, frame_times: np.ndarray) -> np.ndarray:
        """
        Interpolates 2D positions at frame_times, via interpolate_poses (only
        the position output is used -- a dummy identity-quaternion array is
        passed since orientation isn't needed here). Frame times outside
        [timestamps[0], timestamps[-1]] are set to NaN (that robot hasn't
        started, or has already finished, at that point in the video) rather
        than raising, since frame_times spans every robot's combined time
        range, not just this one.

        Args:
            timestamps: Sorted-ascending source timestamps, in seconds.
            xy: (N, 2) source x, y positions.
            frame_times: Target timestamps, in seconds.

        Returns:
            (len(frame_times), 2) interpolated positions, NaN outside the source range.
        """
        in_range = (frame_times >= timestamps[0]) & (frame_times <= timestamps[-1])
        clipped_times = np.clip(frame_times, timestamps[0], timestamps[-1])

        pos = np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1)
        dummy_quat = np.tile([0.0, 0.0, 0.0, 1.0], (len(timestamps), 1))
        new_pos, _ = interpolate_poses(timestamps, pos, dummy_quat, clipped_times)

        xy_interp = new_pos[:, :2]
        xy_interp[~in_range] = np.nan
        return xy_interp

    def _world_to_pixel(self, xy_world: np.ndarray) -> np.ndarray:
        """
        Maps world (x, y) positions to pixel (col, row) coordinates within
        self.background, matching matplotlib's imshow(..., origin="upper")
        convention (row increases downward while y increases upward, hence
        the flip).

        Args:
            xy_world: (N, 2) array of world-space (x, y) positions.

        Returns:
            Array of the same leading shape as xy_world, with pixel (col, row).
        """
        frame_height_px, frame_width_px = self.background.shape[:2]
        x_min, x_max = self.xlim
        y_min, y_max = self.ylim
        col = (xy_world[..., 0] - x_min) / (x_max - x_min) * frame_width_px
        row = (y_max - xy_world[..., 1]) / (y_max - y_min) * frame_height_px
        return np.stack([col, row], axis=-1)

    @staticmethod
    def _cvt_to_pixel_tuple(pt_px: np.ndarray) -> Tuple[int, int]:
        """ Converts a (2,) pixel-space point to an (int, int) tuple, as required by OpenCV. """
        return (int(round(pt_px[0])), int(round(pt_px[1])))

    @staticmethod
    def _draw_constant_trail_segment(canvas: np.ndarray, pt1_px: np.ndarray, pt2_px: np.ndarray,
                                      color_bgr: Tuple[int, int, int], width_px: float) -> None:
        """ Draws one persistent, opaque, anti-aliased line segment directly onto canvas (in place). """
        p1, p2 = VideoGenerator._cvt_to_pixel_tuple(pt1_px), VideoGenerator._cvt_to_pixel_tuple(pt2_px)
        cv2.line(canvas, p1, p2, color_bgr, max(int(round(width_px)), 1), cv2.LINE_AA)

    @staticmethod
    def _decay_fade_alpha(fade_alpha: np.ndarray, decay: float, floor: float) -> None:
        """
        Decays a per-pixel alpha buffer (in place) toward floor. Pixels that
        have never been touched (alpha == 0) are left untouched -- floor only
        applies once a pixel has been drawn on, producing a trail that fades
        near the robot and then stops fading.

        Args:
            fade_alpha: (H, W) alpha buffer, values in [0, 1].
            decay: Multiplicative decay factor per frame, in (0, 1).
            floor: Minimum alpha a touched pixel decays to, in [0, 1).
        """
        if not (0.0 < decay < 1.0):
            raise ValueError("decay must be in (0, 1)")
        if not (0.0 <= floor < 1.0):
            raise ValueError("floor must be in [0, 1)")
        touched = fade_alpha > 0
        fade_alpha[touched] = floor + (fade_alpha[touched] - floor) * decay

    @staticmethod
    def _stamp_fade_trail_segment(fade_alpha: np.ndarray, pt1_px: np.ndarray, pt2_px: np.ndarray, width_px: float) -> None:
        """ Sets alpha to 1.0 (freshly drawn) along a new segment of the fading trail (in place). """
        p1, p2 = VideoGenerator._cvt_to_pixel_tuple(pt1_px), VideoGenerator._cvt_to_pixel_tuple(pt2_px)
        mask = np.zeros(fade_alpha.shape, dtype=np.uint8)
        cv2.line(mask, p1, p2, 255, max(int(round(width_px)), 1), cv2.LINE_AA)
        fade_alpha[mask > 0] = 1.0

    @staticmethod
    def _blend_alpha_layer(frame: np.ndarray, alpha: np.ndarray, color_bgr: Tuple[int, int, int]) -> np.ndarray:
        """ Alpha-blends a solid color onto frame using a per-pixel alpha map in [0, 1]. """
        color_arr = np.array(color_bgr, dtype=np.float32)
        a = alpha[..., None]
        return (frame.astype(np.float32) * (1 - a) + color_arr * a).astype(np.uint8)

    @staticmethod
    def _blend_additive_layer(frame: np.ndarray, alpha: np.ndarray, color_bgr: Tuple[int, int, int]) -> np.ndarray:
        """
        Additively blends a solid color onto frame using a per-pixel alpha map in
        [0, 1] (real bloom, matching the dot's glow halo). Used for the fading
        trail instead of _blend_alpha_layer: a plain alpha blend towards
        color_bgr converges to color_bgr itself, which is a no-op wherever it's
        drawn on top of the constant trail (already that exact color) --
        additive blending brightens those pixels instead, so the fading trail
        reads as a visible highlight even directly over the constant trail.
        """
        color_arr = np.array(color_bgr, dtype=np.float32)
        a = alpha[..., None]
        return np.clip(frame.astype(np.float32) + color_arr * a, 0, 255).astype(np.uint8)

    @staticmethod
    def _blend_lighten_layer(frame: np.ndarray, alpha: np.ndarray, color_bgr: Tuple[int, int, int],
                              lighten_amount: float = 0.4) -> np.ndarray:
        """
        Alpha-blends a lighter tint of color_bgr (blended lighten_amount of the
        way towards white) onto frame using a per-pixel alpha map in [0, 1].
        Used for the fading trail instead of _blend_alpha_layer: blending
        towards color_bgr itself would be a no-op wherever the fade trail is
        drawn on top of the constant trail (already exactly that color), since
        alpha-blending a color into itself always returns that same color --
        blending towards a lighter tint instead makes the fade trail read as a
        highlighted, lighter-colored stretch of that same trail.
        """
        color_arr = np.array(color_bgr, dtype=np.float32)
        white = np.array([255.0, 255.0, 255.0], dtype=np.float32)
        light_color = color_arr + (white - color_arr) * lighten_amount
        a = alpha[..., None]
        return (frame.astype(np.float32) * (1 - a) + light_color * a).astype(np.uint8)

    @staticmethod
    def _perceived_luminance(color_bgr: Tuple[int, int, int]) -> float:
        """
        Returns the perceived brightness of a BGR color via the standard
        luminance formula (0.299 R + 0.587 G + 0.114 B), which weights red and
        green far more heavily than blue -- e.g. a saturated yellow reads as
        much brighter to the eye than a saturated blue of the same magnitude.
        """
        b, g, r = color_bgr
        return 0.299 * r + 0.587 * g + 0.114 * b

    @staticmethod
    def _draw_glow_dot(frame: np.ndarray, center_px: np.ndarray, color_bgr: Tuple[int, int, int],
                        dot_radius_px: float, glow_radius_px: float, glow_target_luminance: float = 90.0) -> np.ndarray:
        """
        Draws a robot's current-position marker: a soft, blurred halo
        (additive blend, i.e. real bloom) plus a solid dot on top.

        The glow's color is scaled so every robot's halo reaches roughly the
        same perceived brightness (glow_target_luminance) regardless of its
        color -- otherwise inherently bright colors (e.g. yellow) blow out
        into a washed-out white halo under additive blending, while inherently
        dim colors (e.g. blue) barely glow at all, even though the blend logic
        is applied identically to both. The solid dot itself keeps the
        original, unscaled color_bgr.
        """

        # Ensure there is something we need to draw
        cx, cy = VideoGenerator._cvt_to_pixel_tuple(center_px)
        h, w = frame.shape[:2]
        margin = glow_radius_px
        if not (-margin <= cx < w + margin and -margin <= cy < h + margin):
            return frame

        # Scale the glow's color towards a common perceived brightness
        luminance = VideoGenerator._perceived_luminance(color_bgr)
        scale = glow_target_luminance / luminance if luminance > 0 else 1.0
        glow_color = tuple(c * scale for c in color_bgr)

        # Draw the glow effect
        glow = np.zeros_like(frame, dtype=np.float32)
        cv2.circle(glow, (cx, cy), max(int(round(glow_radius_px)), 1), glow_color, -1)
        ksize = max(int(round(glow_radius_px)) | 1, 3)
        glow = cv2.GaussianBlur(glow, (ksize, ksize), 0)

        # Merge the glow onto the frame
        frame = np.clip(frame.astype(np.float32) + glow, 0, 255).astype(np.uint8)

        # Draw the dot representing the robot
        cv2.circle(frame, (cx, cy), max(int(round(dot_radius_px)), 1), color_bgr, -1, lineType=cv2.LINE_AA)
        return frame

    def generate(self, timestamps_list: List[np.ndarray], positions_world_list: List[np.ndarray],
                 colors_bgr: List[Tuple[int, int, int]], video_duration_sec: float,
                 names: Optional[List[str]] = None, title: Optional[str] = None) -> None:
        """
        Builds one shared frame-time grid spanning every robot's combined
        timestamp range (sized to video_duration_sec at self.fps), interpolates
        each robot's position onto it, then renders every frame and either
        saves it to self.save_path or plays it back live.

        Args:
            timestamps_list: One (N_r,) array of sorted-ascending timestamps
                (seconds) per robot.
            positions_world_list: One (N_r, 2) array of world-space XY positions
                per robot, aligned with that robot's own timestamps_list entry.
            colors_bgr: One (B, G, R) 0-255 tuple per robot.
            video_duration_sec: Duration of the output video, in seconds. Every
                robot's trajectory is time-warped to fit within this duration.
            names: Robot names, drawn as a color-coded legend in the top-left
                corner if provided.
            title: Optional video title, drawn across the top of every frame.

        Raises:
            ValueError: If timestamps_list, positions_world_list, colors_bgr, and
                (when given) names aren't the same length, or a video writer
                could not be opened.
        """

        # Check arguments
        num_robots = len(timestamps_list)
        if (num_robots != len(positions_world_list) or num_robots != len(colors_bgr)
                or (names is not None and num_robots != len(names))):
            raise ValueError("Lengths of timestamps_list, positions_world_list, colors_bgr, and names must be equal!")

        # Build one shared frame-time grid, and interpolate each robot's position onto it
        time_ranges: List[Tuple[float, float]] = [(float(ts[0]), float(ts[-1])) for ts in timestamps_list]
        frame_times: np.ndarray = self.compute_frame_times(time_ranges, video_duration_sec)
        positions_world: List[np.ndarray] = [self._interpolate_xy_at_times(ts, xy, frame_times)
                                              for ts, xy in zip(timestamps_list, positions_world_list)]

        # Map positions to pixel coordinates
        positions_px: List[np.ndarray] = [self._world_to_pixel(xy) for xy in positions_world]
        num_frames = len(frame_times)
        frame_height_px, frame_width_px = self.background.shape[:2]

        # Create canvas for constant trails and fading trails
        constant_canvas: np.ndarray = self.background.copy()
        fade_alphas: List[np.ndarray] = [np.zeros((frame_height_px, frame_width_px), dtype=np.float32) for _ in range(num_robots)]

        # Keep track of the previous pixel a robot was at
        prev_px: List[Optional[np.ndarray]] = [None] * num_robots

        # Open a video writer (re-encoding to H.264 on exit) if we aren't displaying
        # interactively, via a no-op context manager instead
        live_playback = self.save_path is None
        writer_ctx = nullcontext(None) if live_playback \
            else self.open_video_writer(self.save_path, self.fps, (frame_width_px, frame_height_px))

        # Start drawing frames
        pbar = tqdm.tqdm(total=num_frames, desc="Rendering video...", unit=" frames")
        try:
            with writer_ctx as writer:
                for i in range(num_frames):

                    # Draw the trails
                    for r in range(num_robots):
                        curr_px = positions_px[r][i]
                        curr_px = None if np.any(np.isnan(curr_px)) else curr_px
                        if prev_px[r] is not None and curr_px is not None:
                            self._draw_constant_trail_segment(constant_canvas, prev_px[r], curr_px, colors_bgr[r], self.constant_trail_width_px)
                        self._decay_fade_alpha(fade_alphas[r], self.fade_trail_decay, self.fade_trail_floor)
                        if prev_px[r] is not None and curr_px is not None:
                            self._stamp_fade_trail_segment(fade_alphas[r], prev_px[r], curr_px, self.fade_trail_width_px)
                        # Only advance prev_px on a valid position -- once a robot has
                        # started, letting curr_px's NaN (its trajectory has ended) null
                        # out prev_px would make its dot vanish instead of freezing in place.
                        if curr_px is not None:
                            prev_px[r] = curr_px
                    frame = constant_canvas.copy()
                    for r in range(num_robots):
                        frame = self._blend_lighten_layer(frame, fade_alphas[r], colors_bgr[r])

                    # Draw robot dots
                    for r in range(num_robots):
                        if prev_px[r] is not None:
                            frame = self._draw_glow_dot(frame, prev_px[r], colors_bgr[r], self.dot_radius_px, self.glow_radius_px)

                    # Put legend and title
                    if names is not None:
                        for r, name in enumerate(names):
                            y = 25 + r * 25
                            cv2.circle(frame, (20, y), 6, colors_bgr[r], -1, cv2.LINE_AA)
                            cv2.putText(frame, name, (35, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                    if title is not None:
                        cv2.putText(frame, title, (frame_width_px // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

                    # Write or show frame
                    if writer is not None:
                        writer.write(frame)
                        pbar.update()
                    else:
                        cv2.imshow("VideoGenerator", frame)
                        pbar.update()
                        if cv2.waitKey(max(int(round(1000 / self.fps)), 1)) == 27:
                            break
        finally:
            pbar.close()
            if live_playback:
                cv2.destroyAllWindows()
