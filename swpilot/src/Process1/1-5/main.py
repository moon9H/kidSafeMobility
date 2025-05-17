# 과정 1 - (문제5) "화성 기지의 취약점을 찾아서 보강하자"

import numpy as np

file1 = '1-5/1-5-mars_base_main_parts-001.csv'
file2 = '1-5/1-5-mars_base_main_parts-002.csv'
file3 = '1-5/1-5-mars_base_main_parts-003.csv'
output_file = '1-5/parts_to_work_on.csv'

try :
    csv_data1 = []
    csv_data2 = []
    csv_data3 = []
    header = []
    with open(file1, 'r', encoding='utf-8-sig') as f :
        part, strength = f.readline().strip().split(',')
        header.append([part, strength])
        for line in f :
            part, strength = line.strip().split(',')
            csv_data1.append([part, strength])
        np_csv1 = np.array(csv_data1)
    with open(file2, 'r', encoding='utf-8-sig') as f :
        next(f)
        for line in f :
            part, strength = line.strip().split(',')
            csv_data2.append([part, strength])
        np_csv2 = np.array(csv_data2)
    with open(file3, 'r', encoding='utf-8-sig') as f :
        next(f)
        for line in f :
            part, strength = line.strip().split(',')
            csv_data3.append([part, strength])
        np_csv3 = np.array(csv_data3)
    
    parts = np.concatenate((np_csv1, np_csv2, np_csv3), axis = 1)
    avg_list = []
    for part in parts :
        avg_list.append([str(part[0]), (int(part[1]) + int(part[3]) + int(part[5])) // 3])
    avg_list = sorted(avg_list, key=lambda x: x[1], reverse=True)
    with open(output_file, 'w', encoding='utf-8-sig') as f :
        f.write(header[0][0] + ',' + header[0][1] + '\n')
        for part in avg_list :
            if int(part[1]) < 50 :
                f.write(part[0] + ',' + str(part[1]) + '\n')
    
    # [보너스 과제] parts_to_work_on.csv를 읽어서 parts2라는 ndarray에 저장
    csv_data4 = []
    with open(output_file, 'r', encoding='utf-8-sig') as f :
        next(f)
        for line in f :
            part, strength = line.strip().split(',')
            csv_data4.append([part, strength])
        part2 = np.array(csv_data4)
    
    part3 = part2.T
    print(part3)

except Exception as e:
    print(f"오류 발생: {e}")