# 과정 1 - (문제9) "미션 컴퓨터 모니터링"

import platform
import psutil
import time
import random
import threading
import multiprocessing  # 멀티프로세싱을 위한 모듈

# 더미 센서 클래스
class DummySensor:
    def __init__(self):
        self.env_values = {
            'mars_base_internal_temperature': None,
            'mars_base_external_temperature': None,
            'mars_base_internal_humidity': None,
            'mars_base_external_illuminance': None,
            'mars_base_internal_co2': None,
            'mars_base_internal_oxygen': None
        }
    
    def set_env(self):
        self.env_values['mars_base_internal_temperature'] = random.uniform(18, 30)
        self.env_values['mars_base_external_temperature'] = random.uniform(0, 21)
        self.env_values['mars_base_internal_humidity'] = random.uniform(50, 60)
        self.env_values['mars_base_external_illuminance'] = random.uniform(500, 715)
        self.env_values['mars_base_internal_co2'] = random.uniform(0.02, 0.1)
        self.env_values['mars_base_internal_oxygen'] = random.uniform(4, 7)
    
    def get_env(self):
        return self.env_values

# 미션 컴퓨터 클래스
class MissionComputer:
    def __init__(self):
        self.env_values = {
            'mars_base_internal_temperature': None,
            'mars_base_external_temperature': None,
            'mars_base_internal_humidity': None,
            'mars_base_external_illuminance': None,
            'mars_base_internal_co2': None,
            'mars_base_internal_oxygen': None
        }
        self.ds = DummySensor()
    
    def get_sensor_data(self):
        while True:
            # 센서의 값을 가져와서 env_values에 담음
            self.ds.set_env()
            self.env_values = self.ds.get_env()
            print('Sensor data:')
            print('{')
            for key, value in self.env_values.items():
                print(f'\t"{key}" : "{value:.3f}"')
            # 5초에 한 번씩 반복
            print('}\n')
            time.sleep(5)
    
    def get_mission_computer_info(self):
        try:
            while True:
                # 시스템 정보를 수집
                system_info = {
                    'operating_system': platform.system(),
                    'os_version': platform.release(),
                    'cpu_type': platform.processor(),
                    'cpu_cores': psutil.cpu_count(logical=True),
                    'memory_total': psutil.virtual_memory().total,
                }
                print('Mission Computer 시스템 정보:')
                # JSON 형식으로 문자열을 직접 생성
                json_str = '{\n'
                json_str += f'    "운영체계": "{system_info["operating_system"]}",\n'
                json_str += f'    "운영체계 버전": "{system_info["os_version"]}",\n'
                json_str += f'    "CPU의 타입": "{system_info["cpu_type"]}",\n'
                json_str += f'    "CPU의 코어 수": {system_info["cpu_cores"]},\n'
                json_str += f'    "메모리의 크기": {system_info["memory_total"]}\n'
                json_str += "}\n"

                print(json_str)
                time.sleep(20)  # 20초에 한 번씩 시스템 정보 측정
        
        except Exception as e:
            print(f'시스템 정보를 가져오던 중 오류가 발생했습니다! {str(e)}')

    def get_mission_computer_load(self):
        try:
            while True:
                # 부하 정보를 수집
                load_info = {
                    "cpu_usage": psutil.cpu_percent(interval=1),  # CPU 실시간 사용량
                    "memory_usage": psutil.virtual_memory().percent,  # 메모리 실시간 사용량
                }

                print('\nMission Computer 부하 정보:')

                json_str = "{\n"
                json_str += f'    "CPU 실시간 사용량": {load_info["cpu_usage"]},\n'
                json_str += f'    "메모리 실시간 사용량": {load_info["memory_usage"]}\n'
                json_str += "}\n"

                print(json_str)
                time.sleep(20)  # 20초에 한 번씩 부하 정보 측정
        
        except Exception as e:
            print(f'시스템 부하 정보를 가져오던 중 오류가 발생했습니다! {str(e)}')

# 멀티 프로세싱 실행
def run_mission_computer_info():
    runComputer1 = MissionComputer()
    runComputer1.get_mission_computer_info()

def run_mission_computer_load():
    runComputer2 = MissionComputer()
    runComputer2.get_mission_computer_load()

def run_sensor_data():
    runComputer3 = MissionComputer()
    runComputer3.get_sensor_data()

if __name__ == "__main__":
    # 멀티프로세싱을 사용하여 각 인스턴스 실행
    info_process = multiprocessing.Process(target=run_mission_computer_info)
    load_process = multiprocessing.Process(target=run_mission_computer_load)
    sensor_process = multiprocessing.Process(target=run_sensor_data)

    # 프로세스 시작
    info_process.start()
    load_process.start()
    sensor_process.start()

    # 프로세스가 모두 완료될 때까지 대기
    try : 
        info_process.join()
        load_process.join()
        sensor_process.join()
    # [보너스 과제] CLTR+C 입력 시 출력 종료
    except KeyboardInterrupt :
        info_process.terminate()
        load_process.terminate()
        sensor_process.terminate()
        info_process.join()
        load_process.join()
        sensor_process.join()

