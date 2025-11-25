#!/usr/bin/env python3
"""
stitch_sift.py

Usage:
    python stitch_sift.py /path/to/images_folder -o output.jpg

Dependencies:
    pip install opencv-contrib-python numpy
"""

import os
import argparse
import cv2
import numpy as np


def load_images_from_folder(folder):
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
             if f.lower().endswith(exts)]
    imgs = [cv2.imread(f) for f in files]
    files = [f for f, im in zip(files, imgs) if im is not None]
    imgs = [im for im in imgs if im is not None]
    return files, imgs


def detect_and_compute_sift(img, sift):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, des = sift.detectAndCompute(gray, None)
    return kps, des


def match_descriptors(des1, des2):
    # BFMatcher with default params; use kNN and ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for m_n in knn:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def find_homography_from_matches(kp1, kp2, matches, min_matches=8):
    if len(matches) < min_matches:
        return None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)  # maps pts2 -> pts1
    return H, mask


def compose_global_homographies(images):
    sift = cv2.SIFT_create()
    kps = []
    dess = []
    for im in images:
        kp, des = detect_and_compute_sift(im, sift)
        kps.append(kp)
        dess.append(des)

    # global_h[0] = identity (map image 0 into base coord)
    global_h = [np.eye(3)]
    for i in range(1, len(images)):
        des_prev = dess[i - 1]
        des_cur = dess[i]
        kp_prev = kps[i - 1]
        kp_cur = kps[i]
        if des_prev is None or des_cur is None or len(kp_prev) < 4 or len(kp_cur) < 4:
            print(f"WARNING: not enough features between images {i-1} and {i}; using identity.")
            H_pair = np.eye(3)
        else:
            matches = match_descriptors(des_prev, des_cur)
            pair = find_homography_from_matches(kp_prev, kp_cur, matches)
            if pair is None or pair[0] is None:
                print(f"WARNING: homography failed between images {i-1} and {i}; using identity.")
                H_pair = np.eye(3)
            else:
                H_pair = pair[0]
        # compose: H maps points in image_i to image_{i-1}
        # global for i = global_{i-1} @ H_pair
        H_global = global_h[i - 1] @ H_pair
        # normalize
        H_global = H_global / H_global[2, 2]
        global_h.append(H_global)
    return global_h


def warp_and_blend(images, homographies):
    # compute canvas extents by transforming corners
    corners = []
    for im, H in zip(images, homographies):
        h, w = im.shape[:2]
        pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        pts_h = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        corners.append(pts_h)
    all_pts = np.vstack(corners)
    x_min, y_min = np.floor(all_pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_pts.max(axis=0)).astype(int)

    # translation to shift everything into positive coordinates
    tx = -x_min if x_min < 0 else 0
    ty = -y_min if y_min < 0 else 0
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    print(f"Canvas size: {canvas_w} x {canvas_h}")

    accumulator = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for idx, (im, H) in enumerate(zip(images, homographies)):
        Ht = H.copy()
        # add translation
        T = np.array([[1, 0, tx],
                      [0, 1, ty],
                      [0, 0, 1]], dtype=np.float64)
        Ht = T @ Ht
        warped = cv2.warpPerspective(im, Ht, (canvas_w, canvas_h))
        mask = cv2.warpPerspective(np.ones((im.shape[0], im.shape[1]), dtype=np.uint8), Ht, (canvas_w, canvas_h))
        mask_f = mask.astype(np.float32)

        # accumulate weighted sum (simple averaging where images overlap)
        accumulator += warped.astype(np.float32) * mask_f[:, :, None]
        weight += mask_f

    # avoid divide by zero
    nonzero = weight > 0
    result = np.zeros_like(accumulator, dtype=np.uint8)
    result[nonzero] = (accumulator[nonzero] / weight[nonzero, None]).astype(np.uint8)

    # crop to content bbox
    ys, xs = np.where(weight > 0)
    if len(xs) == 0 or len(ys) == 0:
        return result
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cropped = result[y0:y1 + 1, x0:x1 + 1]
    return cropped


def main():
    parser = argparse.ArgumentParser(description="Stitch images in a folder using SIFT.")
    parser.add_argument("folder", help="Folder containing images (overlapping)")
    parser.add_argument("-o", "--output", default="panorama.jpg", help="Output filename")
    args = parser.parse_args()

    files, images = load_images_from_folder(args.folder)
    if len(images) == 0:
        print("No images found in folder.")
        return
    if len(images) == 1:
        cv2.imwrite(args.output, images[0])
        print("Single image - saved as output.")
        return

    print(f"Loaded {len(images)} images.")
    homographies = compose_global_homographies(images)
    pano = warp_and_blend(images, homographies)
    cv2.imwrite(args.output, pano)
    print(f"Panorama saved to {args.output}")


if __name__ == "__main__":
    main()
