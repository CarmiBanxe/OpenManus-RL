import os
import shutil
from pathlib import Path
from typing import Union

_ALWAYS_SKIP = {".git"}


def merge_repositories(
    repo1_path: Union[str, Path],
    repo2_path: Union[str, Path],
    merged_path: Union[str, Path],
    exclude_git: bool = True,
) -> None:
    """
    Сливает два репозитория в один.

    Args:
        repo1_path: Путь к первому репозиторию
        repo2_path: Путь ко второму репозиторию
        merged_path: Путь к merged репозиторию
        exclude_git: Если True — пропускает .git (сохраняет git-историю repo1)
    """
    repo1_path = Path(repo1_path).expanduser()
    repo2_path = Path(repo2_path).expanduser()
    merged_path = Path(merged_path).expanduser()

    merged_path.mkdir(parents=True, exist_ok=True)

    skip = _ALWAYS_SKIP if exclude_git else set()

    for item in os.listdir(repo1_path):
        if item in skip:
            continue
        src_path = repo1_path / item
        dst_path = merged_path / item

        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)

    for item in os.listdir(repo2_path):
        if item in skip:
            continue
        src_path = repo2_path / item
        dst_path = merged_path / item

        if src_path.is_dir():
            if dst_path.exists():
                merge_directories(src_path, dst_path, skip)
            else:
                shutil.copytree(src_path, dst_path)
        else:
            if dst_path.exists():
                handle_file_conflict(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)


def merge_directories(
    src_dir: Union[str, Path],
    dst_dir: Union[str, Path],
    skip: set[str],
) -> None:
    """Рекурсивно сливает директории"""
    src_dir = Path(src_dir).expanduser()
    dst_dir = Path(dst_dir).expanduser()

    for item in os.listdir(src_dir):
        if item in skip:
            continue
        src_path = src_dir / item
        dst_path = dst_dir / item

        if src_path.is_dir():
            if dst_path.exists():
                merge_directories(src_path, dst_path, skip)
            else:
                shutil.copytree(src_path, dst_path)
        else:
            if dst_path.exists():
                handle_file_conflict(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)


def handle_file_conflict(src_file: Union[str, Path], dst_file: Union[str, Path]) -> None:
    """Обрабатывает конфликты файлов — второй репо побеждает"""
    print(f"Конфликт файлов: {dst_file} будет перезаписан")
    shutil.copy2(src_file, dst_file)


if __name__ == "__main__":
    repo1_path = "~/OpenManus"
    repo2_path = "~/wt/banking-engine-b0b1"
    merged_path = "~/merged-repo"

    # exclude_git=True: .git из repo1 (OpenManus) сохраняется в merged-repo
    # exclude_git=False: .git из repo2 (banking-engine) перезаписывает
    merge_repositories(repo1_path, repo2_path, merged_path, exclude_git=True)
    print(f"Репозитории успешно слиты в {merged_path}")
