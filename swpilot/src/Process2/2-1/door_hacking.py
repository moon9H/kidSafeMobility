import zipfile
import time
import string
import math
import multiprocessing
import sys

# 균등 분할 함수
def divide_charset_evenly(charset, num_chunks):
    charset_list = list(charset)
    total = len(charset_list)
    base = total // num_chunks
    remainder = total % num_chunks

    chunks = []
    start = 0
    for i in range(num_chunks):
        # 앞에서부터 나머지 만큼 1개씩 추가
        chunk_size = base + (1 if i < remainder else 0)
        end = start + chunk_size
        chunks.append(charset_list[start:end])
        start = end
    return chunks

# 비밀번호 탐색 작업자
def try_passwords(start_chars, zip_path, found_flag, result_queue, task_id, progress_queue):
    charset = string.ascii_lowercase + string.digits
    total = len(start_chars) * (len(charset) ** 5)
    counter = 0
    last_report = time.time()

    try:
        with zipfile.ZipFile(zip_path) as zf:
            for a in start_chars:
                for b in charset:
                    for c in charset:
                        for d in charset:
                            for e in charset:
                                for f in charset:
                                    if found_flag.value:
                                        return
                                    pwd = a + b + c + d + e + f
                                    try:
                                        zf.extractall(pwd=bytes(pwd, 'utf-8'))
                                        found_flag.value = True
                                        result_queue.put(pwd)
                                        return
                                    except:
                                        counter += 1
                                        now = time.time()
                                        if now - last_report >= 1:
                                            # 구조화된 메시지 전송
                                            progress_queue.put((task_id, start_chars[0], start_chars[-1], counter, total, pwd))
                                            last_report = now
    except:
        return

# 병렬 해제 실행 함수
def parallel_unlock_zip(zip_path="src/Process2/2-1/2-1-emergency_storage_key.zip"):
    import string, math, sys, time, multiprocessing
    charset = list(string.ascii_lowercase + string.digits)
    num_processes = multiprocessing.cpu_count()

    # 문자 분할 함수 (리스트 기반, 균등 분할)
    def divide_charset_evenly_fixed(charset_list, num_chunks):
        total = len(charset_list)
        base = total // num_chunks
        remainder = total % num_chunks
        chunks = []
        start = 0
        for i in range(num_chunks):
            chunk_size = base + (1 if i < remainder else 0)
            end = start + chunk_size
            chunks.append(charset_list[start:end])
            start = end
        return chunks

    chunks = divide_charset_evenly_fixed(charset, num_processes)
    found_flag = multiprocessing.Value('b', False)
    result_queue = multiprocessing.Queue()
    progress_queue = multiprocessing.Queue()
    processes = []
    latest_messages = [""] * len(chunks)

    start_time = time.time()
    start_fmt = time.strftime('%Y-%m-%d %H:%M:%S')

    print(f"[병렬 시작] {num_processes}개 코어 사용")
    print(f"[시작 시각] {start_fmt}")
    print("-" * 60)

    for i, start_chars in enumerate(chunks):
        p = multiprocessing.Process(
            target=try_passwords,
            args=(start_chars, zip_path, found_flag, result_queue, i, progress_queue)
        )
        processes.append(p)
        p.start()

    try:
        while any(p.is_alive() for p in processes):
            try:
                task_id, start_c, end_c, count, total, pwd = progress_queue.get(timeout=0.5)
                percent = count / total * 100
                msg = f"[P{task_id}] [{start_c}~{end_c}] 진행률: {percent:.2f}% ({count:,}/{total:,}) | 현재: {pwd}"
                latest_messages[task_id] = (msg, count, total)

                # 출력
                sys.stdout.write("\033[H\033[J")  # 터미널 clear
                print(f"[병렬 진행 상태 - {len(chunks)}개 코어]")
                print(f"[시작 시각] {start_fmt}")
                print("-" * 60)

                total_done = 0
                total_tasks = 0
                for entry in latest_messages:
                    if isinstance(entry, tuple):
                        line, c, t = entry
                        print(line)
                        total_done += c
                        total_tasks += t
                    else:
                        print(entry)

                if total_tasks > 0:
                    overall = total_done / total_tasks * 100
                    print("-" * 60)
                    print(f"[전체 진행률] {overall:.2f}% ({total_done:,}/{total_tasks:,})")

            except:
                continue
    except KeyboardInterrupt:
        print("\n[중단됨] 사용자 종료")
    finally:
        for p in processes:
            p.terminate()

    print()
    if not result_queue.empty():
        password = result_queue.get()
        elapsed = time.time() - start_time
        print(f"\n[성공] 비밀번호: {password}")
        print(f"[소요 시간] {elapsed:.2f}초")
        with open("password.txt", "w") as f:
            f.write(password)
    else:
        print("[실패] 병렬 방식으로도 찾지 못했습니다.")
    
# 실행 예시
if __name__ == "__main__":
    parallel_unlock_zip()