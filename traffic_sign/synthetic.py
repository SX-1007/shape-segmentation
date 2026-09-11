"""Deterministic, training-free background-stress dataset generation."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .models import SignShape


_SEED = 0x51A9D42


class _CvRng:
    """Python port of the cv::RNG stream used by the reference generator."""

    _kn: np.ndarray | None = None
    _wn: np.ndarray | None = None
    _fn: np.ndarray | None = None

    def __init__(self, seed: int) -> None:
        self.state = seed if seed else 0xFFFFFFFF
        if self.__class__._kn is None:
            self.__class__._initialize_normal_tables()

    @staticmethod
    def _advance(state: int) -> int:
        return (
            (state & 0xFFFFFFFF) * 4_164_903_690
            + ((state >> 32) & 0xFFFFFFFF)
        ) & 0xFFFFFFFFFFFFFFFF

    def _next(self) -> int:
        self.state = self._advance(self.state)
        return self.state & 0xFFFFFFFF

    def integers(
        self,
        low: int,
        high: int,
        size: int | None = None,
    ) -> int | np.ndarray:
        def one() -> int:
            return self._next() % (high - low) + low

        if size is None:
            return one()
        return np.array([one() for _ in range(size)], dtype=np.int64)

    def uniform(self, low: float, high: float) -> float:
        first = self._next()
        combined = (first << 32) | self._next()
        return combined * 5.421010862427522e-20 * (high - low) + low

    @classmethod
    def _initialize_normal_tables(cls) -> None:
        kn = np.zeros(128, dtype=np.uint32)
        wn = np.zeros(128, dtype=np.float32)
        fn = np.zeros(128, dtype=np.float32)
        m1 = 2_147_483_648.0
        dn = 3.442619855899
        tn = dn
        vn = 9.91256303526217e-3
        q = vn / math.exp(-0.5 * dn * dn)
        kn[0] = int((dn / q) * m1)
        kn[1] = 0
        wn[0] = np.float32(q / m1)
        wn[127] = np.float32(dn / m1)
        fn[0] = np.float32(1.0)
        fn[127] = np.float32(math.exp(-0.5 * dn * dn))
        for index in range(126, 0, -1):
            dn = math.sqrt(-2.0 * math.log(vn / dn + math.exp(-0.5 * dn * dn)))
            kn[index + 1] = int((dn / tn) * m1)
            tn = dn
            fn[index] = np.float32(math.exp(-0.5 * dn * dn))
            wn[index] = np.float32(dn / m1)
        cls._kn, cls._wn, cls._fn = kn, wn, fn

    def normal_int16(self, sigma: float, shape: tuple[int, ...]) -> np.ndarray:
        assert self._kn is not None and self._wn is not None and self._fn is not None
        count = math.prod(shape)
        values = np.empty(count, dtype=np.float32)
        state = self.state
        rng_float = np.float32(2.3283064365386963e-10)
        float_min = np.finfo(np.float32).tiny
        normal_tail = np.float32(3.442620)
        for output_index in range(count):
            while True:
                unsigned_hz = state & 0xFFFFFFFF
                hz = unsigned_hz if unsigned_hz < 0x80000000 else unsigned_hz - 0x100000000
                state = self._advance(state)
                strip = hz & 127
                x = np.float32(np.float32(hz) * self._wn[strip])
                if abs(hz) < int(self._kn[strip]):
                    break
                if strip == 0:
                    while True:
                        x_tail = np.float32((state & 0xFFFFFFFF) * rng_float)
                        state = self._advance(state)
                        y_tail = np.float32((state & 0xFFFFFFFF) * rng_float)
                        state = self._advance(state)
                        x_tail = np.float32(
                            -math.log(float(np.float32(x_tail + float_min))) * 0.2904764
                        )
                        y_tail = np.float32(-math.log(float(np.float32(y_tail + float_min))))
                        if float(y_tail + y_tail) >= float(x_tail * x_tail):
                            break
                    x = np.float32(normal_tail + x_tail if hz > 0 else -normal_tail - x_tail)
                    break
                y = np.float32((state & 0xFFFFFFFF) * rng_float)
                state = self._advance(state)
                left = np.float32(
                    self._fn[strip]
                    + y * np.float32(self._fn[strip - 1] - self._fn[strip])
                )
                if float(left) < math.exp(-0.5 * float(x) * float(x)):
                    break
            values[output_index] = x
        self.state = state
        scaled = np.rint(values * np.float32(sigma))
        return np.clip(scaled, -32768, 32767).astype(np.int16).reshape(shape)


def _draw_background(
    scene: int,
    rng: _CvRng,
) -> np.ndarray:
    gradients = (
        ((45, 86, 42), (31, 58, 25)),
        ((80, 105, 142), (55, 74, 106)),
        ((205, 175, 125), (105, 100, 92)),
        ((125, 125, 125), (68, 73, 78)),
        ((75, 64, 54), (28, 30, 32)),
        ((48, 62, 83), (92, 109, 130)),
        ((150, 150, 145), (78, 82, 86)),
        ((28, 32, 35), (8, 10, 14)),
    )
    top, bottom = gradients[scene % 8]
    weights = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None, None]
    image = np.rint(
        (1.0 - weights) * np.array(top, np.float32)[None, None, :]
        + weights * np.array(bottom, np.float32)[None, None, :]
    ).astype(np.uint8)
    image = np.repeat(image, 256, axis=1)

    if scene % 8 == 0:
        for _ in range(180):
            center_y = int(rng.integers(0, 256))
            center_x = int(rng.integers(0, 256))
            center = (center_x, center_y)
            radius = int(rng.integers(2, 15))
            red = int(rng.integers(18, 70))
            green = int(rng.integers(55, 145))
            blue = int(rng.integers(20, 75))
            color = (blue, green, red)
            cv2.circle(image, center, radius, color, cv2.FILLED, cv2.LINE_AA)
    elif scene % 8 == 1:
        for y in range(12, 256, 24):
            cv2.line(image, (0, y), (255, y), (155, 165, 170), 2)
        for row, y in enumerate(range(0, 256, 24)):
            for x in range(20 if row & 1 else 0, 256, 40):
                cv2.line(image, (x, y), (x, min(255, y + 24)), (145, 150, 156), 2)
    elif scene % 8 == 2:
        x = -10
        while x < 266:
            roof = int(rng.integers(95, 180))
            width = int(rng.integers(25, 55))
            left = max(0, x)
            rectangle_width = min(width, 256 - left)
            red = int(rng.integers(55, 115))
            green = int(rng.integers(55, 115))
            blue = int(rng.integers(55, 115))
            color = (blue, green, red)
            if rectangle_width > 0:
                cv2.rectangle(
                    image,
                    (left, roof, rectangle_width, 256 - roof),
                    color,
                    cv2.FILLED,
                )
            for window_y in range(roof + 9, 245, 17):
                for window_x in range(left + 6, min(256, x + width - 4), 13):
                    cv2.rectangle(
                        image,
                        (window_x, window_y, 6, 7),
                        (175, 160, 100),
                        cv2.FILLED,
                    )
            x += int(rng.integers(24, 43))
    else:
        for index in range(55):
            start_y = int(rng.integers(0, 256))
            start_x = int(rng.integers(-20, 256))
            start = (start_x, start_y)
            delta_y = int(rng.integers(4, 45))
            delta_x = int(rng.integers(8, 70))
            end = (start_x + delta_x, start_y + delta_y)
            if index % 11 == 0:
                color = (35, 35, 185)
            elif index % 13 == 0:
                color = (170, 75, 35)
            elif index % 17 == 0:
                color = (30, 190, 205)
            else:
                red = int(rng.integers(30, 180))
                green = int(rng.integers(30, 180))
                blue = int(rng.integers(30, 180))
                color = (blue, green, red)
            cv2.rectangle(image, start, end, color, cv2.FILLED, cv2.LINE_AA)
        for _ in range(18):
            thickness = int(rng.integers(1, 4))
            red = int(rng.integers(25, 210))
            green = int(rng.integers(25, 210))
            blue = int(rng.integers(25, 210))
            end_y = int(rng.integers(0, 256))
            end_x = int(rng.integers(0, 256))
            start_y = int(rng.integers(0, 256))
            start_x = int(rng.integers(0, 256))
            cv2.line(
                image,
                (start_x, start_y),
                (end_x, end_y),
                (blue, green, red),
                thickness,
                cv2.LINE_AA,
            )
    return image


def _draw_synthetic_sign(shape: SignShape) -> tuple[np.ndarray, np.ndarray]:
    sign = np.zeros((160, 160, 3), dtype=np.uint8)
    mask = np.zeros((160, 160), dtype=np.uint8)
    red, blue, yellow = (35, 35, 205), (190, 82, 28), (35, 205, 225)
    white, dark = (225, 230, 230), (22, 25, 28)
    if shape is SignShape.CIRCLE:
        cv2.circle(mask, (80, 80), 58, 255, cv2.FILLED, cv2.LINE_8)
        cv2.circle(sign, (80, 80), 58, red, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(sign, (80, 80), 47, white, cv2.FILLED, cv2.LINE_AA)
        cv2.putText(sign, "50", (48, 96), cv2.FONT_HERSHEY_DUPLEX, 1.35, dark, 3, cv2.LINE_AA)
    elif shape is SignShape.TRIANGLE:
        outer = np.array([[80, 16], [19, 137], [141, 137]], np.int32)
        inner = np.array([[80, 31], [36, 125], [124, 125]], np.int32)
        cv2.fillConvexPoly(mask, outer, 255, cv2.LINE_8)
        cv2.fillConvexPoly(sign, outer, dark, cv2.LINE_AA)
        cv2.fillConvexPoly(sign, inner, yellow, cv2.LINE_AA)
        cv2.line(sign, (79, 61), (79, 100), dark, 6, cv2.LINE_AA)
        cv2.circle(sign, (79, 113), 4, dark, cv2.FILLED, cv2.LINE_AA)
    elif shape is SignShape.RECTANGLE:
        outer = np.rint(cv2.boxPoints(((80.0, 80.0), (112.0, 92.0), 0.0))).astype(np.int32)
        cv2.fillConvexPoly(mask, outer, 255, cv2.LINE_8)
        cv2.fillConvexPoly(sign, outer, blue, cv2.LINE_AA)
        cv2.arrowedLine(sign, (45, 82), (114, 82), white, 10, cv2.LINE_AA, 0, 0.32)
    else:
        octagon = np.array(
            [
                [
                    round(80 + 60 * math.cos(math.pi / 8.0 + index * math.pi / 4.0)),
                    round(80 + 60 * math.sin(math.pi / 8.0 + index * math.pi / 4.0)),
                ]
                for index in range(8)
            ],
            np.int32,
        )
        cv2.fillConvexPoly(mask, octagon, 255, cv2.LINE_8)
        cv2.fillConvexPoly(sign, octagon, red, cv2.LINE_AA)
        cv2.putText(sign, "STOP", (37, 89), cv2.FONT_HERSHEY_DUPLEX, 0.83, white, 3, cv2.LINE_AA)
    return sign, mask


def _composite_perspective(
    background: np.ndarray,
    shape: SignShape,
    sample: int,
    rng: _CvRng,
) -> tuple[np.ndarray, np.ndarray]:
    sign, source_mask = _draw_synthetic_sign(shape)
    half_width = float(rng.uniform(52.0, 76.0))
    half_height = half_width * float(rng.uniform(0.72, 1.05))
    angle = math.radians(float(rng.uniform(-24.0, 24.0)))
    center_y = float(rng.integers(78, 178))
    center_x = float(rng.integers(82, 175))
    center = (center_x, center_y)
    source = np.array([[12, 12], [148, 12], [148, 148], [12, 148]], np.float32)
    destination = []
    for sign_x, sign_y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        x = sign_x * half_width
        y = sign_y * half_height
        rotated_x = x * math.cos(angle) - y * math.sin(angle)
        rotated_y = x * math.sin(angle) + y * math.cos(angle)
        jitter_y = float(rng.uniform(-7.0, 7.0))
        jitter_x = float(rng.uniform(-7.0, 7.0))
        destination.append(
            (center[0] + rotated_x + jitter_x, center[1] + rotated_y + jitter_y)
        )
    transform = cv2.getPerspectiveTransform(source, np.array(destination, np.float32))
    warped_sign = cv2.warpPerspective(
        sign,
        transform,
        (256, 256),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    truth = cv2.warpPerspective(
        source_mask,
        transform,
        (256, 256),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    cv2.copyTo(warped_sign, truth, background)

    if sample % 4 == 0:
        shadow = np.array(
            [
                [0, int(rng.integers(70, 150))],
                [255, int(rng.integers(125, 215))],
                [255, 255],
                [0, 255],
            ],
            np.int32,
        )
        shadow_mask = np.zeros((256, 256), np.uint8)
        cv2.fillConvexPoly(shadow_mask, shadow, 145, cv2.LINE_8)
        shade = np.full_like(background, 35)
        blended = cv2.addWeighted(background, 0.68, shade, 0.32, 0.0)
        cv2.copyTo(blended, shadow_mask, background)
    if sample % 5 == 0:
        background = cv2.GaussianBlur(background, (5, 5), 1.15)
    if sample % 3 == 0:
        noise = rng.normal_int16(9.0, background.shape)
        background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if sample % 7 == 0:
        success, encoded = cv2.imencode(".jpg", background, [cv2.IMWRITE_JPEG_QUALITY, 52])
        if success:
            background = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return background, truth


def generate_synthetic_background_dataset(
    root: Path | str,
    image_count: int = 64,
) -> bool:
    destination = Path(root)
    image_count = max(4, int(image_count))
    images_dir = destination / "images"
    masks_dir = destination / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    shapes = (
        SignShape.CIRCLE,
        SignShape.TRIANGLE,
        SignShape.RECTANGLE,
        SignShape.OCTAGON,
    )
    rng = _CvRng(_SEED)
    label_lines: list[str] = []
    metadata_lines = ["file,shape,background,seed"]
    for index in range(image_count):
        shape = shapes[index % 4]
        scene = (index // 4) % 8
        image = _draw_background(scene, rng)
        image, mask = _composite_perspective(image, shape, index, rng)
        name = f"synthetic_{index:03d}.png"
        if not cv2.imwrite(str(images_dir / name), image):
            return False
        if not cv2.imwrite(str(masks_dir / name), mask):
            return False
        label_lines.append(f"{name} {shape.name}")
        metadata_lines.append(f"{name},{shape.name},{scene},0x51A9D42")
    (destination / "labels.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    (destination / "metadata.csv").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    return True
