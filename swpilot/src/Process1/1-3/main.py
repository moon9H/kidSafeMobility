# 과정 1 - (문제3) "인화 물질을 찾아라"

filepath = "1-3/1-3-Mars_Base_Inventory_List.csv"
savepath = "1-3/Mars_Base_Inventory_danger.csv"
bin_path = "1-3/Mars_Base_Inventory_List.bin"

lines = []

# 파일 읽기 및 리스트 객체에 저장
try:
    f = open(filepath, 'r', encoding='utf-8')
    line_num = 0
    while True:
        line = f.readline()
        if line == "":
            break
        lines.append(line.strip())
        line_num += 1
    f.close()
except Exception as e:
    print(f"오류 발생: {e}")

# 컬럼 별로 배열 형태로 저장
data = [["" for _ in range(5)] for _ in range(line_num - 1)]
header = []
row_count = 0

for i in range(len(lines)):
    cols = lines[i].split(",")
    if i == 0:
        header = cols
    else:
        for j in range(5):
            data[row_count][j] = cols[j]
        row_count += 1

# 인화성 지수 기준 정렬 (내림차순, 버블 정렬)
for i in range(row_count):
    for j in range(i + 1, row_count):
        if float(data[i][4]) < float(data[j][4]):
            temp = data[i]
            data[i] = data[j]
            data[j] = temp

# 인화성 ≥ 0.7 출력
print("인화성 0.7 이상 화물:")
danger_rows = []
for i in range(row_count):
    if float(data[i][4]) >= 0.7:
        line = ",".join(data[i])
        print(line)
        danger_rows.append(line)

# 인화성 ≥ 0.7 CSV 저장
try:
    f = open(savepath, "w", encoding='utf-8')
    f.write(",".join(header) + "\n")
    for line in danger_rows:
        f.write(line + "\n")
    f.close()
except Exception as e:
    print(f"오류 발생: {e}")

# [보너스 과제] 이진 파일로 저장
try:
    f = open(bin_path, "wb")
    for i in range(row_count):
        line = ",".join(data[i]) + "\n"
        f.write(line.encode("utf-8"))
    f.close()
except Exception as e:
    print(f"오류 발생: {e}")

# 이진 파일 읽기 및 출력
try:
    print("\n------------인화성 순서로 정렬된 내용 출력------------")
    f = open(bin_path, "rb")
    while True:
        byte_line = f.readline()
        if byte_line == b"":
            break
        print(byte_line.decode("utf-8").strip())
    f.close()
except Exception as e:
    print(f"오류 발생: {e}")