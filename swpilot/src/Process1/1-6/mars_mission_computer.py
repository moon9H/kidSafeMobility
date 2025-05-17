# 과정 1 - (문제6) "미션 컴퓨터 리턴즈"

import random
# 보너스 과제를 위해 사용함.
import os
from datetime import datetime

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
        # [보너스 과제] 출력하는 내용 파일에 log를 남기는 부분을 get_env()에 추가 한다.
        log_path = "1-6/env.log"
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

# 인스턴스 생성 및 테스트
ds = DummySensor()
ds.set_env()
print(ds.get_env())