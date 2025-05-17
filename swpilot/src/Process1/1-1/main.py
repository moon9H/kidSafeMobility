# 과정 1 - (문제1) "컴퓨터를 복구하고 사고 원인을 파악해보자"

from datetime import datetime

print("Hello Mars")

filepath = '/Users/hwangmungyu/Desktop/swpilot/Process1/1-1/1-1-mission_computer_main.log'
reportpath = '/Users/hwangmungyu/Desktop/swpilot/Process1/1-1/log_analysis.md'
errorpath = '/Users/hwangmungyu/Desktop/swpilot/Process1/1-1/error.log'
full_log = []

cutoff_time = datetime.strptime("2023-08-27 11:35:00", "%Y-%m-%d %H:%M:%S")


critical_events = []
# 문제 부분 저장용 리스트 (Bonus 과제)
error_log = []

try:
    with open(filepath, 'r', encoding='utf-8') as f:
        for content in f :
            print(content,end='')
            full_log.append(content)
            if content.startswith('timestamp') :
                error_log.append(content)
                continue
            timestamp_str = content.split(",")[0]
            log_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            if log_time >= cutoff_time :
                error_log.append(content)
            if any(keyword in content.lower() for keyword in ['unstable', 'explosion', 'critical']):
                    critical_events.append(content.strip())
            
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다")

except Exception as e:
    print(f"오류 발생: {e}")

# Markdown 보고서 자동 생성
try:
    with open(reportpath, 'w', encoding='utf-8') as f:
        f.write("# Mission Log Analysis Report\n\n")
        f.write("## 사고 개요\n")
        f.write("발사 이후 시스템에서 이상 징후가 발생했습니다.\n\n")

        f.write("## 주요 이상 로그\n")
        if critical_events:
            for line in critical_events:
                f.write(f"- {line}\n")
        else:
            f.write("- 분석된 이상 로그가 없습니다.\n")

        f.write("\n## 사고 원인 분석\n")
        if any("oxygen tank" in line.lower() and "explosion" in line.lower() for line in critical_events):
            f.write("- 추정 원인: 산소 탱크의 불안정 및 폭발\n")
        else:
            f.write("- 명확한 사고 원인을 로그에서 추정할 수 없음\n")

except Exception as e:
    print(f"보고서 작성 중 오류 발생: {e}")

print("----------------Reversed Log Part----------------")
# 역순 출력 파트 (Bonus 과제)
for i in range(len(full_log) - 1, -1, -1) :
    print(full_log[i],end='')

# 문제 부분 파일 저장 (Bonus 과제)
try:
    with open(errorpath, "w", encoding="utf-8") as outfile:
        outfile.writelines(error_log)
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다")
except Exception as e:
    print(f"오류 발생: {e}")

f.close()