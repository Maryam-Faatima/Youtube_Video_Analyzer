from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw


def run_dip(pil_img: Image.Image) -> dict[str, Image.Image | float]:
    arr = np.array(pil_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    results: dict[str, Image.Image | float] = {}
    results["1. Original"] = pil_img
    results["2. Grayscale"] = Image.fromarray(gray)
    results["3. Gaussian Blur"] = Image.fromarray(blurred)

    edges = cv2.Canny(blurred, 40, 120)
    results["4. Canny Edges"] = Image.fromarray(edges)

    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sx ** 2 + sy ** 2)
    sobel = np.uint8(np.clip(sobel / (sobel.max() + 1e-8) * 255, 0, 255))
    results["5. Sobel Gradient"] = Image.fromarray(sobel)

    lap = cv2.Laplacian(blurred, cv2.CV_64F)
    lap = np.uint8(np.clip(np.abs(lap) / (np.abs(lap).max() + 1e-8) * 255, 0, 255))
    results["6. Laplacian"] = Image.fromarray(lap)
    results["7. Histogram EQ"] = Image.fromarray(cv2.equalizeHist(gray))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    results["8. CLAHE"] = Image.fromarray(clahe.apply(gray))

    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    results["9. Adaptive Threshold"] = Image.fromarray(thresh)

    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    results["10. Morphological Close"] = Image.fromarray(morph)

    corners = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    corner_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    corner_vis[corners > 0.01 * corners.max()] = [255, 0, 0]
    results["11. Harris Corners"] = Image.fromarray(corner_vis)

    dft = np.fft.fftshift(np.fft.fft2(gray))
    spectrum = 20 * np.log(np.abs(dft) + 1)
    spectrum = np.uint8(np.clip((spectrum - spectrum.min()) / (spectrum.ptp() + 1e-8) * 255, 0, 255))
    results["12. DFT Spectrum"] = Image.fromarray(spectrum)

    pixels = arr.reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(np.float32(pixels), 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(arr.shape)
    results["13. K-Means"] = Image.fromarray(segmented)

    edge_density = float(np.count_nonzero(edges)) / float(edges.size or 1)
    focus_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    results["_edge_density"] = edge_density
    results["_focus_score"] = focus_score
    return results


def make_dip_grid(pil_img: Image.Image) -> Image.Image:
    dip = run_dip(pil_img)
    tiles = [
        dip["1. Original"],
        dip["2. Grayscale"],
        dip["3. Gaussian Blur"],
        dip["4. Canny Edges"],
        dip["5. Sobel Gradient"],
        dip["6. Laplacian"],
        dip["7. Histogram EQ"],
        dip["8. CLAHE"],
        dip["9. Adaptive Threshold"],
        dip["10. Morphological Close"],
        dip["11. Harris Corners"],
        dip["12. DFT Spectrum"],
        dip["13. K-Means"],
    ]

    labels = [
        "Original",
        "Grayscale",
        "Gaussian Blur",
        "Canny Edges",
        "Sobel Gradient",
        "Laplacian",
        "Histogram EQ",
        "CLAHE",
        "Adaptive Threshold",
        "Morph Close",
        "Harris Corners",
        "DFT Spectrum",
        "K-Means",
    ]

    cols = 3
    thumb_w, thumb_h, label_h = 240, 135, 24
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (thumb_w * cols, (thumb_h + label_h) * rows), (255, 235, 245))
    draw = ImageDraw.Draw(grid)

    for index, (tile, label) in enumerate(zip(tiles, labels)):
        col, row = index % cols, index // cols
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        grid.paste(tile.resize((thumb_w, thumb_h), Image.LANCZOS), (x, y))
        draw.rectangle([x, y + thumb_h, x + thumb_w, y + thumb_h + label_h], fill=(255, 210, 230))
        draw.text((x + 6, y + thumb_h + 5), label, fill=(136, 0, 68))

    return grid
