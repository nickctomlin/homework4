from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    from .generate_qa import extract_kart_objects, extract_track_info
    import json

    captions = []

    # Extract kart objects and track info
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    # Get ego kart name from info file (ego is always track_id 0)
    with open(info_path) as f:
        info = json.load(f)
    ego_kart_name = info["karts"][0]  # track_id 0 is always ego

    # Find other karts
    other_karts = [k for k in kart_objects if not k["is_ego"]]

    # Image center for relative position calculations
    image_center_x = img_width / 2
    image_center_y = img_height / 2

    # 1. Ego car caption
    captions.append(f"{ego_kart_name} is the ego car.")

    # 2. Counting caption
    num_visible_karts = len(kart_objects)
    captions.append(f"There are {num_visible_karts} karts in the scene.")

    # 3. Track name caption
    captions.append(f"The track is {track_name}.")

    # 4. Relative position captions for each non-ego kart
    for kart in other_karts:
        kart_name = kart["kart_name"]
        cx, cy = kart["center"]

        # Left/Right relative to image center
        if cx < image_center_x:
            lr_position = "left"
        else:
            lr_position = "right"

        # Front/Behind - lower y = higher in image = in front
        if cy < image_center_y:
            fb_position = "in front"
        else:
            fb_position = "behind"

        # Add position caption
        captions.append(f"{kart_name} is {lr_position} of the ego car.")
        captions.append(f"{kart_name} is {fb_position} of the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate_all_captions(
    data_dir: str = "supertux_data/data/train",
    output_file: str = "data/train/all_captions.json",
    image_prefix: str = "train",
):
    """
    Generate captions for all info files in the data directory.

    Args:
        data_dir: Directory containing the info.json files and images
        output_file: Output JSON file path
        image_prefix: Prefix for image file paths in output (e.g., 'train')
    """
    import json

    data_path = Path(data_dir)
    info_files = sorted(data_path.glob("*_info.json"))

    all_captions = []

    print(f"Found {len(info_files)} info files in {data_dir}")

    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")

        # Process all 10 views (0-9)
        for view_index in range(10):
            # Check if corresponding image exists
            image_file = data_path / f"{base_name}_{view_index:02d}_im.jpg"
            if not image_file.exists():
                continue

            try:
                captions = generate_caption(str(info_file), view_index)

                # Add each caption as a separate entry (relative to data directory)
                relative_image_path = f"{image_prefix}/{base_name}_{view_index:02d}_im.jpg"
                for caption in captions:
                    all_captions.append({
                        "image_file": relative_image_path,
                        "caption": caption,
                    })
            except Exception as e:
                print(f"Error processing {info_file} view {view_index}: {e}")
                continue

    print(f"Generated {len(all_captions)} captions")

    # Write to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_captions, f, indent=2)

    print(f"Saved captions to {output_file}")
    print(f"\nNOTE: Make sure to copy/symlink images from '{data_dir}' to 'data/{image_prefix}/' before training.")


"""
Usage Example: Visualize captions for a specific file and view:
   python -m homework.generate_captions check --info_file data/valid/00000_info.json --view_index 0

Generate all captions:
   python -m homework.generate_captions generate --data_dir data/train --output_file data/train/all_captions.json
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate_all_captions})


if __name__ == "__main__":
    main()
