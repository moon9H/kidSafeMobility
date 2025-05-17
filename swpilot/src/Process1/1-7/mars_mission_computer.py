# 과정 1 - (문제7) "살아난 미션 컴퓨터"

import random
import time
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
                # [보너스 과제] 5분마다 평균 출력
                if count == 60:
                    avg_result = {}
                    for key in self.env_values.keys():
                        values = [float(item[key]) for item in avg_buffer]
                        avg_result[key] = round(sum(values) / len(values), 3)
                    print("5분 평균값:", avg_result)
                    avg_buffer = []
                    count = 0
                time.sleep(5)
        # [보너스 과제] 특정 키 입력 시 출력 중지
        except KeyboardInterrupt:
            print("System stopped....")

RunComputer = MissionComputer()
RunComputer.get_sensor_data()