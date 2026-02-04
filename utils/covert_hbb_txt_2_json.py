import json
from pathlib import Path
from PIL import Image
import yaml


def load_class_names(classes_file: str = None) -> dict:
    """
    从 classes.txt 或 classes.yaml 加载类别名称

    Args:
        classes_file: 类别文件路径（可选）

    Returns:
        类别 ID 到名称的映射字典
    """
    class_names = {}

    if not classes_file:
        return class_names

    classes_path = Path(classes_file)

    # 尝试读取 classes.txt
    classes_txt = (
        classes_path
        if classes_path.name.endswith(".txt")
        else classes_path / "classes.txt"
    )
    if classes_txt.exists():
        try:
            with open(classes_txt, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        class_names[idx] = line
            print(f"📋 已加载 {len(class_names)} 个类别从 classes.txt")
            return class_names
        except Exception as e:
            print(f"⚠️  读取 classes.txt 失败: {e}")

    # 尝试读取 classes.yaml
    classes_yaml = (
        classes_path
        if classes_path.name.endswith(".yaml")
        else classes_path / "classes.yaml"
    )
    if classes_yaml.exists():
        try:
            with open(classes_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if "names" in data:
                    class_names = data["names"]
                    print(f"📋 已加载 {len(class_names)} 个类别从 classes.yaml")
                    return class_names
        except Exception as e:
            print(f"⚠️  读取 classes.yaml 失败: {e}")

    return class_names


def convert_hbb_txt_to_labelimg_json(
    images_dir: str, labels_dir: str, output_dir: str = None, classes_file: str = None
) -> None:
    """
    将 YOLO Hbb txt 格式转换为 LabelImg JSON 格式
    每个 txt 文件转换为对应的 JSON 文件

    Args:
        images_dir: 图片文件夹路径
        labels_dir: txt 标注文件夹路径
        output_dir: 输出 JSON 文件夹路径（默认与 labels 相同）
        classes_file: 类别文件路径（默认为 labels_dir/classes.txt）
    """

    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir) if output_dir else labels_path

    if not images_path.exists():
        raise FileNotFoundError(f"图片文件夹不存在: {images_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"标注文件夹不存在: {labels_dir}")

    output_path.mkdir(parents=True, exist_ok=True)

    # 加载类别名称
    if classes_file is None:
        classes_file = str(labels_path / "classes.txt")
    class_names = load_class_names(classes_file)

    # 获取所有 txt 文件（排除 classes.txt）
    txt_files = sorted(
        [f for f in labels_path.glob("*.txt") if f.name != "classes.txt"]
    )
    total_annotations = 0

    for txt_file in txt_files:
        # 找对应的图片文件
        image_name = txt_file.stem
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_file = None

        for ext in image_extensions:
            potential_image = images_path / f"{image_name}{ext}"
            if potential_image.exists():
                image_file = potential_image
                break

        if not image_file:
            print(f"⚠️  找不到图片对应 {txt_file.name}")
            continue

        # 获取图片信息
        try:
            img = Image.open(image_file)
            img_width, img_height = img.size
        except Exception as e:
            print(f"❌ 无法读取图片 {image_file}: {e}")
            continue

        # 构建相对路径
        try:
            image_rel_path = f"../../images/train/{image_file.name}"
        except Exception:
            image_rel_path = f"../../images/train/{image_file.name}"

        # 初始化 LabelImg JSON 结构
        labelimg_json = {
            "version": "3.3.9",
            "flags": {},
            "shapes": [],
            "imagePath": image_rel_path,
            "imageData": None,
            "imageHeight": img_height,
            "imageWidth": img_width,
        }

        # 读取 txt 标注
        try:
            with open(txt_file, "r") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"❌ 无法读取标注文件 {txt_file}: {e}")
            continue

        # 解析每一行标注
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                print(f"⚠️  标注行格式错误 {txt_file.name}: {line}")
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # 将归一化坐标转换回像素坐标
                x_min = (x_center - width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                x_max = (x_center + width / 2) * img_width
                y_max = (y_center + height / 2) * img_height

                # 确保坐标在有效范围内
                x_min = max(0, min(x_min, img_width))
                y_min = max(0, min(y_min, img_height))
                x_max = max(0, min(x_max, img_width))
                y_max = max(0, min(y_max, img_height))

                # 获取类别名称
                label_name = class_names.get(class_id, f"{class_id:03d}")

                # 创建矩形 shape
                shape = {
                    "label": label_name,
                    "score": None,
                    "points": [
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max],
                    ],
                    "group_id": None,
                    "description": "",
                    "difficult": False,
                    "shape_type": "rectangle",
                    "flags": {},
                    "attributes": {},
                    "kie_linking": [],
                }

                labelimg_json["shapes"].append(shape)
                total_annotations += 1

            except ValueError as e:
                print(f"⚠️  无法解析标注数据 {txt_file.name}: {line}")
                continue

        # 保存 JSON 文件
        json_output_file = output_path / f"{image_name}.json"
        try:
            with open(json_output_file, "w", encoding="utf-8") as f:
                json.dump(labelimg_json, f, indent=2, ensure_ascii=False)
            print(
                f"✅ {txt_file.name} -> {json_output_file.name} ({len(labelimg_json['shapes'])} 个框)"
            )
        except Exception as e:
            print(f"❌ 无法保存 JSON 文件 {json_output_file}: {e}")

    # 打印统计信息
    print(f"\n{'=' * 50}")
    print(f"✅ 转换完成!")
    print(f"📊 统计信息:")
    print(f"   - 处理文件数: {len(txt_files)}")
    print(f"   - 总标注数: {total_annotations}")
    print(f"💾 输出文件夹: {output_path.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="将 YOLO Hbb txt 标注转换为 LabelImg JSON 格式"
    )
    parser.add_argument("--images", type=str, required=True, help="图片文件夹路径")
    parser.add_argument("--labels", type=str, required=True, help="txt 标注文件夹路径")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件夹路径（默认与 labels 相同）",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="类别文件路径（默认为 labels/classes.txt）",
    )

    args = parser.parse_args()

    convert_hbb_txt_to_labelimg_json(
        args.images, args.labels, args.output, args.classes
    )
