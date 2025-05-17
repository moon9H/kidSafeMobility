# 과정 1 - (문제8) "불안정한 미션 컴퓨터..."

import random
import time
import platform
import psutil

# 로그 파일 저장을 위해 사용되는 라이브러리
from datetime import datetime
import os

class DummySensor:
    def __init__(self):
        self.env_values = {
            "mars_base_internal_temperature": None,
            "mars_base_external_temperature": None,
            "mars_base_internal_humidity": None,
            "mars_base_external_illuminance": None,
            "mars_base_internal_co2": None,
            "mars_base_internal_oxygen": None
        }

    def set_env(self):
        self.env_values["mars_base_internal_temperature"] = round(random.uniform(18, 30), 2)
        self.env_values["mars_base_external_temperature"] = round(random.uniform(0, 21), 2)
        self.env_values["mars_base_internal_humidity"] = round(random.uniform(50, 60), 2)
        self.env_values["mars_base_external_illuminance"] = round(random.uniform(500, 715), 2)
        self.env_values["mars_base_internal_co2"] = round(random.uniform(0.02, 0.1), 4)
        self.env_values["mars_base_internal_oxygen"] = round(random.uniform(4, 7), 2)

    def get_env(self):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = (
            f"{now},"
            f"{self.env_values['mars_base_internal_temperature']},"
            f"{self.env_values['mars_base_external_temperature']},"
            f"{self.env_values['mars_base_internal_humidity']},"
            f"{self.env_values['mars_base_external_illuminance']},"
            f"{self.env_values['mars_base_internal_co2']},"
            f"{self.env_values['mars_base_internal_oxygen']}\n"
        )
        log_path = "1-7/env.log"
        header = "datetime,mars_base_internal_temperature,mars_base_external_temperature,mars_base_internal_humidity,mars_base_external_illuminance,mars_base_internal_co2,mars_base_internal_oxygen\n"
        try:
            file_exists = os.path.exists(log_path)
            with open(log_path, "a", encoding="utf-8") as f:
                if not file_exists:
                    f.write(header)
                f.write(log_line)
        except Exception as e:
            print(f"파일 기록 중 오류 발생: {e}")
        return self.env_values

class MissionComputer(DummySensor):
    def __init__(self):
        super().__init__()
        self.system_info = {
            'operating_system': platform.system(),
            'os_version': platform.release(),
            'cpu_type': platform.processor(),
            'cpu_cores': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total
        }
        self.load_info = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent
        }
        self.setting_keys = self.load_setting_keys()
    
    def get_sensor_data(self):
        print("센서 데이터 출력 중... (중지하려면 Ctrl+C)")
        avg_buffer = []
        count = 0
        try:
            while True:
                self.set_env()
                env = self.get_env()
                avg_buffer.append(env.copy())
                # JSON 형태로 출력
                json_str = '{\n\t' + ',\n\t'.join([f'"{k}": {repr(v)}' for k, v in env.items()]) + '\n}'
                print(json_str)
                count += 1
                # 5분마다 평균 출력
                if count == 60:
                    avg_result = {}
                    for key in self.env_values.keys():
                        values = [float(item[key]) for item in avg_buffer]
                        avg_result[key] = round(sum(values) / len(values), 3)
                    print("5분 평균값:", avg_result)
                    avg_buffer = []
                    count = 0
                time.sleep(5)
        # 특정 키 입력 시 출력 중지
        except KeyboardInterrupt:
            print("System stopped....")
    
    def get_mission_computer_info(self):
        try :
            # 시스템 정보를 수집
            system_info = {
                'operating_system': platform.system(),
                'os_version': platform.release(),
                'cpu_type': platform.processor(),
                'cpu_cores': psutil.cpu_count(logical=True),  # 논리적 코어 수
                'memory_total': psutil.virtual_memory().total  # 메모리 총 용량
            }

            # JSON 형식으로 문자열을 직접 생성
            json_str = '{\n'
            json_str += f'\t"운영체계": "{system_info["operating_system"]}",\n'
            json_str += f'\t"운영체계 버전": "{system_info["os_version"]}",\n'
            json_str += f'\t"CPU의 타입": "{system_info["cpu_type"]}",\n'
            json_str += f'\t"CPU의 코어 수": {system_info["cpu_cores"]},\n'
            json_str += f'\t"메모리의 크기": {system_info["memory_total"]}\n'
            json_str += "}"

            return json_str
        
        except Exception as e:
            return f'시스템 정보를 가져오던 중 오류가 발생했습니다 ! {str(e)}'

    def get_mission_computer_load(self):
        try :
            # 부하 정보를 수집
            load_info = {
                "cpu_usage": psutil.cpu_percent(interval=1),  # CPU 실시간 사용량 (1초 간격)
                "memory_usage": psutil.virtual_memory().percent  # 메모리 실시간 사용량
            }

            # JSON 형식으로 문자열을 직접 생성
            json_str = "{\n"
            json_str += f'\t"CPU 실시간 사용량": {load_info["cpu_usage"]},\n'
            json_str += f'\t"메모리 실시간 사용량": {load_info["memory_usage"]}\n'
            json_str += "}"

            return json_str

        except Exception as e:
            return f'시스템 정보를 가져오던 중 오류가 발생했습니다 ! {str(e)}'
    

    # [보너스 과제] setting.txt 파일을 만들어서 출력되는 정보의 항목을 셋팅 할 수 있도록 코드를 수정한다.
    def load_setting_keys(self):
        setting_path = "1-8/setting.txt"
        all_items = list(self.system_info.items()) + list(self.load_info.items())
        setting_dir = os.path.dirname(setting_path)
        if setting_dir and not os.path.exists(setting_dir):
            os.makedirs(setting_dir)
        if not os.path.exists(setting_path):
            with open(setting_path, "w", encoding="utf-8") as f:
                for k, v in all_items:
                    f.write(f"{k}:{v}\n")
            return [f"{k}:{v}" for k, v in all_items]
        else:
            with open(setting_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            return lines if lines else [f"{k}:{v}" for k, v in all_items]

# MissionComputer 클래스의 인스턴스화
runComputer = MissionComputer()

# 시스템 정보 확인
print('Mission Computer의 시스템 정보 :')
print(runComputer.get_mission_computer_info())

# 부하 정보 확인
print('\nMission Computer의 부하:')
print(runComputer.get_mission_computer_load())

runComputer.load_setting_keys()