"""文档素材同步工具：把 ``outputs/*.png`` 同步到 ``docs/assets/``。

设计约束
--------

1. 增量同步：只有当图片内容发生变化时才覆盖，避免每次跑 ``main.py`` 都
   触发 Git 的"未变更文件时间戳更新"。
2. 孤儿检测：目标目录中在来源目录没有同名文件的素材会被标记为
   ``[孤儿]``；默认仅告警，``--prune`` 才删除。
3. 检查模式：``--check`` 只做一致性校验，发现不同步时返回 1，不写入任何
   文件。适合 CI。
4. 双入口：既可以直接 ``python sync_docs.py``，也支持
   ``from sync_docs import sync_outputs_to_docs`` 被 ``main.py`` 调用。

返回约定
--------
* ``sync_outputs_to_docs`` 返回 ``SyncReport`` 数据类，含新增/更新/跳过/孤儿
  的数量与路径。
* ``__main__`` 入口根据 ``--check`` 与同步结果返回相应的系统退出码
  （``--check`` 模式下不同步则退出 1；正常模式下不抛出则退出 0）。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


#: 默认来源目录（相对脚本运行目录）。
SOURCE_DIR: str = "outputs"

#: 默认目标目录（相对脚本运行目录）。
TARGET_DIR: str = "docs/assets"

#: 文件块大小（字节）。
CHUNK_SIZE: int = 8192

#: 支持的图片扩展名（小写）。
IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".svg"})


@dataclass
class SyncReport:
    """``sync_outputs_to_docs`` 的同步报告。

    Attributes:
        source_dir: 来源目录绝对路径。
        target_dir: 目标目录绝对路径。
        added: 本次新增的文件列表（目标侧路径）。
        updated: 本次覆盖更新的文件列表。
        unchanged: 内容一致、跳过拷贝的文件列表。
        orphaned: 目标侧存在但来源侧没有的素材文件列表（``--prune`` 模式
            下已被删除的文件会出现在 ``pruned`` 而非 ``orphaned`` 中）。
        pruned: ``--prune`` 模式下删除的孤儿文件列表。
        errors: 处理过程中遇到的错误信息列表（不会因此中断）。
    """

    source_dir: Path
    target_dir: Path
    added: List[Path] = field(default_factory=list)
    updated: List[Path] = field(default_factory=list)
    unchanged: List[Path] = field(default_factory=list)
    orphaned: List[Path] = field(default_factory=list)
    pruned: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def n_changed(self) -> int:
        """新增 + 更新的总数。"""
        return len(self.added) + len(self.updated)

    @property
    def n_total_images(self) -> int:
        """目标目录最终持有的图片总数（不含孤儿）。"""
        return len(self.added) + len(self.updated) + len(self.unchanged)

    def summary(self) -> str:
        """生成一行中文汇总。"""
        return (
            f"[docs] 已同步 {self.n_changed} 张图到 {self.target_dir}: "
            f"新增 {len(self.added)}, 更新 {len(self.updated)}, "
            f"跳过 {len(self.unchanged)}, 孤儿 {len(self.orphaned)}"
        )


def _file_hash(path: Path, algorithm: str = "sha256") -> str:
    """计算文件哈希（分块读取，避免大图片占用过多内存）。

    Args:
        path: 待计算文件路径。
        algorithm: 哈希算法名称，``"md5"`` 或 ``"sha256"``。

    Returns:
        str: 十六进制哈希字符串。
    """
    hasher = hashlib.new(algorithm)
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_image(path: Path) -> bool:
    """判断路径是否为受支持的图片文件。"""
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _collect_images(directory: Path) -> Dict[str, Path]:
    """收集目录下的图片文件，以相对路径为键。

    Args:
        directory: 扫描目录。

    Returns:
        Dict[str, Path]: ``相对路径 -> 绝对路径`` 的映射。
    """
    out: Dict[str, Path] = {}
    if not directory.exists():
        return out
    for p in directory.rglob("*"):
        if _is_image(p):
            out[str(p.relative_to(directory))] = p
    return out


def sync_outputs_to_docs(
    source_dir: str = SOURCE_DIR,
    target_dir: str = TARGET_DIR,
    prune: bool = False,
    check: bool = False,
    algorithm: str = "sha256",
    image_suffixes: Optional[Sequence[str]] = None,
) -> SyncReport:
    """``outputs/*.png`` → ``docs/assets/`` 同步入口。

    Args:
        source_dir: 来源目录路径。
        target_dir: 目标目录路径。
        prune: ``True`` 时删除目标侧孤儿素材；``False`` 时仅标记并告警。
        check: ``True`` 时只做检查，不实际写入/删除文件。
            返回报告中的 ``added``/``updated``/``pruned`` 表示"将要"执行的
            操作，但文件系统保持不变。
        algorithm: 哈希算法，``"md5"`` 或 ``"sha256"``。
        image_suffixes: 覆盖默认图片扩展名集合。

    Returns:
        SyncReport: 同步报告。
    """
    global IMAGE_SUFFIXES
    if image_suffixes is not None:
        IMAGE_SUFFIXES = frozenset(s.lower() for s in image_suffixes)

    src = Path(source_dir).resolve()
    dst = Path(target_dir).resolve()
    report = SyncReport(source_dir=src, target_dir=dst)

    if not src.exists():
        report.errors.append(f"来源目录不存在：{src}")
        return report

    if not check:
        dst.mkdir(parents=True, exist_ok=True)

    src_files = _collect_images(src)
    dst_files = _collect_images(dst)

    # 1. 同步来源侧文件
    for rel, spath in sorted(src_files.items()):
        tpath = dst / rel
        if tpath in (v for v in dst_files.values()):
            # 同名文件已存在，比较哈希
            try:
                src_hash = _file_hash(spath, algorithm)
                dst_hash = _file_hash(tpath, algorithm)
                if src_hash == dst_hash:
                    report.unchanged.append(tpath)
                    print(f"[跳过] {rel} (哈希一致)")
                    continue
                if check:
                    report.updated.append(tpath)
                    print(f"[需更新] {rel}")
                    continue
                shutil.copy2(spath, tpath)
                report.updated.append(tpath)
                print(f"[更新] {rel}")
            except OSError as exc:
                report.errors.append(f"{rel}: {exc}")
        else:
            if check:
                report.added.append(tpath)
                print(f"[需新增] {rel}")
                continue
            try:
                tpath.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(spath, tpath)
                report.added.append(tpath)
                print(f"[新增] {rel}")
            except OSError as exc:
                report.errors.append(f"{rel}: {exc}")

    # 2. 孤儿检测
    for rel, tpath in sorted(dst_files.items()):
        if rel not in src_files:
            report.orphaned.append(tpath)
            if prune:
                if check:
                    report.pruned.append(tpath)
                    print(f"[需删除] 孤儿 {rel}")
                else:
                    try:
                        tpath.unlink()
                        report.pruned.append(tpath)
                        print(f"[删除] 孤儿 {rel}")
                    except OSError as exc:
                        report.errors.append(f"删除 {rel} 失败: {exc}")
            else:
                print(f"[孤儿] {rel} (目标侧存在但来源侧没有；使用 --prune 删除)")

    print(f"\n{report.summary()}")
    return report


def _build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="sync_docs.py",
        description="将 outputs/ 目录下的图片同步到 docs/assets/，"
                    "用于把 main.py 生成的图表推送到文档站点。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python sync_docs.py           # 增量同步\n"
               "  python sync_docs.py --prune   # 同步并删除目标侧孤儿\n"
               "  python sync_docs.py --check   # 只检查，不写入\n",
    )
    parser.add_argument(
        "--source", "-s", default=SOURCE_DIR,
        help=f"来源目录（默认：{SOURCE_DIR}）",
    )
    parser.add_argument(
        "--target", "-t", default=TARGET_DIR,
        help=f"目标目录（默认：{TARGET_DIR}）",
    )
    parser.add_argument(
        "--prune", action="store_true",
        help="删除目标侧存在但来源侧没有的图片（默认仅标记为孤儿）",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="检查模式：只报告差异，不拷贝/删除文件；发现不同步时退出码 1",
    )
    parser.add_argument(
        "--algorithm", choices=("md5", "sha256"), default="sha256",
        help="哈希算法（默认：sha256）",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python sync_docs.py`` 入口。

    Args:
        argv: 命令行参数；``None`` 时使用 ``sys.argv``。

    Returns:
        int: ``--check`` 模式下发现需要同步的内容返回 1，否则返回 0；
        发生错误也返回 1。
    """
    parser = _build_argparser()
    args = parser.parse_args(argv)

    report = sync_outputs_to_docs(
        source_dir=args.source,
        target_dir=args.target,
        prune=args.prune,
        check=args.check,
        algorithm=args.algorithm,
    )

    if report.errors:
        for err in report.errors:
            print(f"[错误] {err}", file=sys.stderr)
        return 1

    if args.check and report.n_changed > 0:
        print("\n[check] 发现不同步内容，请运行同步。", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "SOURCE_DIR",
    "TARGET_DIR",
    "SyncReport",
    "sync_outputs_to_docs",
]
