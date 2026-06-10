import os
import subprocess
import sys

# 配置
BATCH_SIZE_LIMIT = 1000 * 1024 * 1024  # 1000 MB
MAX_FILE_SIZE = 100 * 1024 * 1024     # 100 MB
REMOTE_NAME = "origin"
BRANCH_NAME = "main"
MAX_FILES_PER_BATCH = 5000
AUTO_IGNORE_LARGE_FILES = True  # 新增：超大文件自动加入 .gitignore 

def run_command(command):
    """运行 Shell 命令并返回结果""" 
    try:
        result = subprocess.run(
            command, 
            check=True, 
            shell=False, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        # 忽略 git commit 在没有东西提交时的报错
        if "nothing to commit" in e.stdout or "nothing to commit" in e.stderr:
            return True, "Nothing to commit"
        print(f"Error executing {' '.join(command)}:")
        print(e.stderr)
        return False, e.stderr

def get_changed_files():
    """获取所有变动文件（含 staged/unstaged/untracked/deleted）"""
    print("正在分析 Git 状态，寻找变动文件...")

    # 关键：-uall 展开所有 untracked 文件；-z 用 NUL 分隔避免空格路径解析错误
    cmd = ["git", "status", "--porcelain=v1", "-uall", "-z"]
    success, output = run_command(cmd)
    if not success:
        print("无法获取 git status，请确保在 git 仓库根目录运行。")
        sys.exit(1)

    file_list = []
    # NUL 分隔
    entries = [e for e in output.split("\0") if e]
    for e in entries:
        if len(e) < 4:
            continue
        status = e[:2]         # 例如 "??", " M", "D "
        filepath = e[3:]       # 跳过 "XY "
        filepath = filepath.strip().strip('"')
        if filepath:
            file_list.append((status, filepath))

    return file_list

def get_size(filepath):
    """递归计算文件或目录的总大小"""
    if os.path.isfile(filepath):
        try:
            return os.path.getsize(filepath)
        except OSError:
            return 0
    elif os.path.isdir(filepath):
        total = 0
        for root, dirs, files in os.walk(filepath):
            for file in files:
                try:
                    total += os.path.getsize(os.path.join(root, file))
                except OSError:
                    pass
        return total
    return 0

def append_to_gitignore(filepath: str) -> bool:
    """将路径写入 .gitignore（已存在则不重复），返回是否有变更"""
    gitignore_path = ".gitignore"
    norm = filepath.replace(os.sep, "/").strip()

    existing = set()
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    existing.add(s)

    if norm in existing:
        return False

    with open(gitignore_path, "a", encoding="utf-8") as f:
        f.write(f"\n{norm}\n")
    return True

def main():
    # === 修改点：只获取变动的文件 ===
    files_to_process = get_changed_files()
    
    if not files_to_process:
        print("没有检测到需要提交的文件。")
        return

    print(f"检测到 {len(files_to_process)} 个文件/目录发生变动。")

    files_to_commit = []
    current_batch_size = 0
    batch_count = 1

    for status, filepath in files_to_process:
        # 计算大小
        if os.path.isfile(filepath):
            try:
                file_size = os.path.getsize(filepath)
            except OSError:
                continue

            if file_size >= MAX_FILE_SIZE:
                print(f"⚠️  警告: 文件 {filepath} ({file_size/1024/1024:.2f} MB) 超过 GitHub 100MB 限制，跳过。")
                if AUTO_IGNORE_LARGE_FILES:
                    changed = append_to_gitignore(filepath)
                    if changed:
                        print(f"已加入 .gitignore: {filepath}")
                        # 确保 .gitignore 会被提交
                        files_to_commit.append((" M", ".gitignore"))
                continue
        else:
            file_size = 0

        files_to_commit.append((status, filepath))
        current_batch_size += file_size

        if current_batch_size >= BATCH_SIZE_LIMIT or len(files_to_commit) >= MAX_FILES_PER_BATCH:
            process_batch(files_to_commit, batch_count, current_batch_size)
            files_to_commit = []
            current_batch_size = 0
            batch_count += 1

    # 处理剩余的文件
    if files_to_commit:
        process_batch(files_to_commit, batch_count, current_batch_size)

    print("\n✅ 所有批次处理完成！")

def process_batch(file_list, batch_num, batch_size):
    size_mb = batch_size / (1024 * 1024)
    print(f"\n--- 处理第 {batch_num} 批 (文件数: {len(file_list)}, 大小: {size_mb:.2f} MB) ---")
    chunk_size = 50

    # 区分 add 和 rm
    to_add = [f for s, f in file_list if "D" not in s]
    to_rm = [f for s, f in file_list if "D" in s]

    if to_add:
        print("正在添加到暂存区 (git add)...")
        for i in range(0, len(to_add), chunk_size):
            chunk = to_add[i:i + chunk_size]
            cmd = ["git", "add"] + chunk
            success, _ = run_command(cmd)
            if not success:
                sys.exit(1)

    if to_rm:
        print("正在标记删除 (git rm)...")
        for i in range(0, len(to_rm), chunk_size):
            chunk = to_rm[i:i + chunk_size]
            cmd = ["git", "rm"] + chunk
            success, _ = run_command(cmd)
            if not success:
                sys.exit(1)

    # commit 和 push 保持原样
    print("正在提交 (git commit)...")
    commit_msg = f"Batch upload part {batch_num}"
    cmd = ["git", "commit", "-m", commit_msg]
    success, output = run_command(cmd)
    if "Nothing to commit" in output:
        print("没有检测到更改，跳过推送。")
        return

    print("正在推送 (git push)...")
    cmd = ["git", "push", REMOTE_NAME, BRANCH_NAME]
    success, err = run_command(cmd)
    if success:
        print(f"第 {batch_num} 批推送成功。")
    else:
        print("推送失败。请检查网络或冲突。")
        sys.exit(1)

if __name__ == "__main__":
    main()