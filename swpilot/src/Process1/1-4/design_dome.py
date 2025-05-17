# 과정 1 - (문제4) "화성 기지의 돔을 복구하라"

globalMaterial = ""
globalDiameter = 0
globalThickness = 1
globalArea = 0.0
globalWeight = 0.0

# 화성 중력 계수 (지구 중력의 약 0.38배)
MARS_GRAVITY_RATIO = 0.38

# 밀도 (g/cm³)
MATERIAL_DENSITY = {
    "glass": 2.4,
    "aluminum": 2.7,
    "carbon": 7.85
}

def sphere_area(diameter, material="glass", thickness=1):
    global globalMaterial, globalDiameter, globalThickness, globalArea, globalWeight

    radius_cm = (diameter * 100) / 2  
    area = 2 * 3.14 * (radius_cm ** 2)  
    volume = area * thickness

    density = MATERIAL_DENSITY[material]  # g/cm³
    weight_grams = volume * density  # g
    weight_kg = (weight_grams / 1000) * MARS_GRAVITY_RATIO  # 화성 기준 무게 (kg)

    # 전역 변수 업데이트
    globalMaterial = material
    globalDiameter = diameter
    globalThickness = thickness
    globalArea = round(area, 3)
    globalWeight = round(weight_kg, 3)

    print(f"\n재질 ⇒ {globalMaterial}, 지름 ⇒ {globalDiameter}, 두께 ⇒ {globalThickness}, 면적 ⇒ {globalArea}, 무게 ⇒ {globalWeight} kg")

print("\n--- 반구형 돔 무게 계산기 ---")
try:
    d_input = input("지름을 입력하세요 (m): ")

    diameter = float(d_input)
    if diameter == 0:
        print("지름은 0이 될 수 없습니다.")
        exit()

    m_input = input("재질을 입력하세요 (glass/aluminum/carbon) [기본: glass]: ").strip()
    if m_input == "":
        m_input = "glass"

    t_input = input("두께를 입력하세요 (cm) [기본: 1]: ").strip()
    if t_input == "":
        thickness = 1
    else:
        thickness = float(t_input)

    sphere_area(diameter, m_input, thickness)

# [보너스 과제] 파라메터에 숫자가 아닌 문자가 들어갔을 때 오류가 발생하지 않도록 처리
except ValueError:
    print("숫자로 변환할 수 없는 값이 입력되었습니다. 다시 시도해주세요.")
except:
    print("알 수 없는 오류가 발생했습니다. 다시 시도해주세요.")
