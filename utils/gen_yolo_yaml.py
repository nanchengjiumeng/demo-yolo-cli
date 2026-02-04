from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Union

import yaml


def load_class_names(classes_file: str) -> List[str]:
    """
    从 classes.txt 或 classes.yaml 加载类别名称，返回有序列表。
    """
    classes_path = Path(classes_file)
    if not classes_path.exists():
        raise FileNotFoundError(f"类别文件不存在: {classes_file}")

    if classes_path.suffix.lower() in {".yaml", ".yml"}:
        with open(classes_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        names = data.get("names", {})
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys())]
        if isinstance(names, list):
            return names
        raise ValueError("classes.yaml 中的 names 字段格式不正确")

    if classes_path.suffix.lower() == ".txt":
        with open(classes_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        return [line for line in lines if line]

    raise ValueError("仅支持 .txt 或 .yaml/.yml 类别文件")


def infer_train_val(images_dir: str, base_path: Path) -> Dict[str, str]:
    """
    根据 images_dir 推断 train/val 路径。
    若存在 train/val 子目录则使用它们，否则 train/val 均指向 images_dir。
    """
    images_path = Path(images_dir)
    train_dir = images_path / "train"
    val_dir = images_path / "val"

    if train_dir.exists() and val_dir.exists():
        return {
            "train": str(train_dir.relative_to(base_path)),
            "val": str(val_dir.relative_to(base_path)),
        }

    return {
        "train": str(images_path.relative_to(base_path)),
        "val": str(images_path.relative_to(base_path)),
    }


def generate_yolo_yaml(
    images_dir: str, labels_dir: str, classes_file: str, output_yaml: str
) -> None:
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    if not images_path.exists():
        raise FileNotFoundError(f"images 目录不存在: {images_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels 目录不存在: {labels_dir}")

    names = load_class_names(classes_file)
    base_path = images_path.parent
    train_val = infer_train_val(images_dir, base_path)

    data = {
        "path": str(base_path),
        "train": train_val["train"],
        "val": train_val["val"],
        "labels": str(labels_path.relative_to(base_path)),
        "names": {i: name for i, name in enumerate(names)},
    }

    output_path = Path(output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    print(f"✅ 已生成 YAML: {output_path}")
    print(f"   - train: {data['train']}")
    print(f"   - val: {data['val']}")
    print(f"   - labels: {data['labels']}")
    print(f"   - classes: {len(names)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据 images/labels/classes 生成 YOLO 数据集 YAML"
    )
    parser.add_argument("--images", required=True, help="images 目录路径")
    parser.add_argument("--labels", required=True, help="labels 目录路径")
    parser.add_argument("--classes", required=True, help="classes.txt 或 classes.yaml")
    parser.add_argument(
        "--output",
        default="dataset.yaml",
        help="输出 YAML 路径（默认: dataset.yaml）",
    )

    args = parser.parse_args()
    generate_yolo_yaml(args.images, args.labels, args.classes, args.output)


if __name__ == "__main__":
    main()
