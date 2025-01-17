import subprocess
from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any
import numpy as np
from tqdm import tqdm

IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540


def parse_keypoints(
    keypoints: list[float], occluded: list[float], image_width: int, image_height: int
) -> tuple[list[float], np.ndarray]:
    keypoints_np = np.array(keypoints)
    valid_keypoints_mask = keypoints_np != None
    keypoints_np[np.logical_not(valid_keypoints_mask)] = 0
    keypoints_np = keypoints_np.astype(np.float32)
    keypoints_np = keypoints_np.reshape(-1, 2)
    keypoints_np[:, 0] *= image_width
    keypoints_np[:, 1] *= image_height

    # keypoints_np[1:, :] = 0
    # keypoints_np[1:, :] = 0

    # Z axis for coco seems to be 2 for visible, 1 for occluded, 0 for not annotaded
    # occluded is 1 for occluded, 0 for visible, None for not annotated
    occluded_np = np.array(occluded)
    occluded_np[occluded_np == 0] = 2
    occluded_np[occluded_np == None] = 0

    keypoints_np = np.concatenate([keypoints_np, occluded_np[:, np.newaxis]], axis=1)
    return keypoints_np.flatten().tolist(), keypoints_np, valid_keypoints_mask


def get_bbox(
    keypoints_np: np.ndarray, valid_keypoints_mask: np.ndarray
) -> tuple[float, float, float, float]:
    keypoints_valid = keypoints_np[:, :2].flatten()[valid_keypoints_mask]

    if keypoints_valid.size == 0:
        return (0, 0, 0, 0)

    # keypoints_valid = keypoints_valid.astype(np.float32)
    keypoints_valid = keypoints_valid.reshape(-1, 2)
    x_min = np.min(keypoints_valid[:, 0])
    x_max = np.max(keypoints_valid[:, 0])
    y_min = np.min(keypoints_valid[:, 1])
    y_max = np.max(keypoints_valid[:, 1])
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def main():
    parser = ArgumentParser()
    parser.add_argument("--netid", type=str)
    args = parser.parse_args()

    raw_conflab_path = Path("/tudelft.net/staff-bulk/ewi/insy/SPCDataSets/conflab-mm/")
    if not raw_conflab_path.exists():
        staff_bulk_path = Path("/tudelft.net/staff-bulk")
        staff_bulk_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            "sshfs",
            "-o",
            "ro",
            f"{args.netid}@sftp.tudelft.nl:/staff-bulk/",
            staff_bulk_path,
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            raise RuntimeError("Failed to mount staff-bulk")

    stuff_umbrella_path = Path("/tudelft.net/staff-umbrella")
    processed_conflab_path = stuff_umbrella_path / "neon/experiments/VIT003/"
    if not processed_conflab_path.exists():
        stuff_umbrella_path.mkdir(exist_ok=True, parents=True)
        cmd = [
            "sshfs",
            f"{args.netid}@sftp.tudelft.nl:/staff-umbrella",
            stuff_umbrella_path,
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            raise RuntimeError("Failed to mount staff-umbrella")

    annotations_path = raw_conflab_path / "release/annotations/pose/coco/"

    video_segments_path = raw_conflab_path / "processed/annotation/videoSegments/"

    categories = [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints":
            #    ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],
            [
                "head",
                "nose",
                "neck",
                "rightShoulder",
                "rightElbow",
                "rightWrist",
                "leftShoulder",
                "leftElbow",
                "leftWrist",
                "rightHip",
                "rightKnee",
                "rightAnkle",
                "leftHip",
                "leftKnee",
                "leftAnkle",
                "rightFoot",
                "leftFoot",
            ],
            "skeleton":
            #    [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
            [
                [0, 1],
                [0, 2],
                [2, 3],
                [2, 6],
                [3, 4],
                [4, 5],
                [6, 7],
                [7, 8],
                [2, 9],
                [9, 10],
                [10, 11],
                [11, 15],
                [2, 12],
                [12, 13],
                [13, 14],
                [14, 16],
            ],
        }
    ]

    categories[0]["skeleton"] = (np.array(categories[0]["skeleton"]) + 1).tolist()

    processed_annotations: dict[str, list[dict, Any]] = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    image_path = processed_conflab_path / "images"
    image_path.mkdir(parents=True, exist_ok=True)
    output_annot_path = processed_conflab_path / "annotations" / "person_keypoints.json"

    total_images = 0
    j = 0
    for annotation_file in tqdm(
        sorted(annotations_path.glob("*.json")), desc="Video segment"
    ):
        # Break early
        print(annotation_file.stem)
        if annotation_file.stem != "cam2_vid2_seg9_coco":
            continue

        cam, vid, seg, _ = annotation_file.name.split("_")

        segment_path = video_segments_path / cam / f"{vid}-{seg}-scaled-denoised.mp4"

        if not segment_path.exists():
            raise FileNotFoundError(f"Could not find {segment_path}")

        # Only process two seconds of video, put before the start_number: "-t", str(2),
        # cmd = ["ffmpeg", "-y", "-i", str(segment_path), "-t", str(1), "-start_number", str(total_images), image_path / f"%09d.jpg"]
        # ret = subprocess.run(cmd, capture_output=True)
        # if ret.returncode != 0:
        #     raise RuntimeError(f"Failed to split segment in frames {segment_path}")

        new_images_start = total_images
        total_images = len(list(image_path.glob("*.jpg")))

        with annotation_file.open() as f:
            conflab_annot = json.load(f)

        for i in range(new_images_start, total_images):
            processed_annotations["images"].append(
                {
                    "id": i,
                    "file_name": f"{i:09d}.jpg",
                    "width": IMAGE_WIDTH,
                    "height": IMAGE_HEIGHT,
                }
            )

        multi_people_annot: dict
        for multi_people_annot in tqdm(
            conflab_annot["annotations"]["skeletons"],
            desc="Annot multiple",
            leave=False,
        ):
            for annot in multi_people_annot.values():
                if annot["image_id"] not in [0, 30, 50]:
                    continue
                annot["id"] = j
                annot["image_id"] += new_images_start
                annot["keypoints"], keypoints, valid_keypoints_mask = parse_keypoints(
                    annot["keypoints"],
                    annot["occluded"],
                    image_width=IMAGE_WIDTH,
                    image_height=IMAGE_HEIGHT,
                )
                annot["bbox"] = get_bbox(keypoints, valid_keypoints_mask)
                processed_annotations["annotations"].append(annot)
                j += 1

    output_annot_path.parent.mkdir(parents=True, exist_ok=True)
    with output_annot_path.open("w") as f:
        json.dump(processed_annotations, f, indent=4)


if __name__ == "__main__":
    main()
