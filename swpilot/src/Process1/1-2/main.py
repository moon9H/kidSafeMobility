# 과정 1 - (문제2) "미션 컴퓨터의 내용을 좀 더 분석해 보자"

filepath = '/Users/hwangmungyu/Desktop/swpilot/Process1/1-1/1-1-mission_computer_main.log'
json_save_path = '/Users/hwangmungyu/Desktop/swpilot/Process1/1-2/mission_computer_main.json'

full_log = []

# 콤마를 기준 분류 후 리스트 객체 전환
try:
    f = open(filepath, 'r', encoding='utf-8')
    for line in f:
        if line.startswith('timestamp') :
            continue
        parts = line.strip().split(',', maxsplit=2)
        if len(parts) == 3 :
            timestamp = parts[0]
            event = parts[1]
            msg = parts[2]
            full_log.append([timestamp, event, msg])
    f.close()
except Exception as e:
    print(f"오류 발생: {e}")

# 리스트 객체 출력
for element in full_log :
    print(element)

# 시간 기준 역순 정렬 (단순 문자열 기준)
for i in range(len(full_log)):
    for j in range(i + 1, len(full_log)):
        if full_log[i][0] < full_log[j][0]:
            temp = full_log[i]
            full_log[i] = full_log[j]
            full_log[j] = temp

# 딕셔너리 객체로 전환
log_dict = {}
for i in range(len(full_log)):
    key = "log_" + str(i + 1)
    log_dict[key] = {
        "timestamp": full_log[i][0],
        "event": full_log[i][1],
        "message": full_log[i][2]
    }

# 딕셔너리 객체를 json 파일 포맷으로 저장
json_lines = []
json_lines.append("{")
keys = list(log_dict.keys())
for i in range(len(keys)):
    key = keys[i]
    entry = log_dict[key]
    line = '  "' + key + '": {\n'
    line += '    "timestamp": "' + entry["timestamp"] + '",\n'
    line += '    "event": "' + entry["event"] + '",\n'
    line += '    "message": "' + entry["message"].replace('"', '\\"') + '"\n'
    line += '  }'
    if i < len(keys) - 1:
        line += ","
    json_lines.append(line)
json_lines.append("}")

try:
    f = open(json_save_path, 'w', encoding='utf-8')
    for line in json_lines:
        f.write(line + "\n")
    f.close()
except Exception as e:
    print(f"오류 발생: {e}")

# 보너스 과제 : 특정 문자열 검색 시 해당 내용 출력
keyword = input("검색 키워드 입력 (ex: Oxygen): ").lower()

print(f"'{keyword}'가 포함된 로그:")
found = False
for key in log_dict:
    msg = log_dict[key]["message"].lower()
    if keyword in msg:
        print(f"{key}: {log_dict[key]}")
        found = True

if not found:
    print("해당 키워드가 포함된 로그를 찾을 수 없습니다.")