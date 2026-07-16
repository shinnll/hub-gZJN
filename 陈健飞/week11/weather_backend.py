"""
weather_backend.py — 天气查询后端（三种方式共享的业务逻辑）

教学重点：
  1. 同样是"纯业务逻辑"，与 rag_backend 平级，被三种方式复用
  2. 内部两次 HTTP 请求：Geocoding（城市名→经纬度）+ 天气查询
  3. 错误处理返回可读字符串而非抛异常，方便 LLM 直接消费

使用方式（作为模块）：
  from src.weather_backend import get_weather
  print(get_weather("宁德"))

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
"""

import httpx
from dataclasses import dataclass

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo 天气代码 → 中文描述映射
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


@dataclass
class CityDimension:
    name: str
    latitude: float
    longitude: float
    country: str = ""
    admin1: str = ""
    feature_code: str = ""
    population: int = 0

    def get_weather(self) -> str:
        return get_weather_from_dimension(self)


# ↓↓↓ 类结束。下面 geocode_city 顶格写，是模块级函数 ↓↓↓
def geocode_city(city: str) -> "CityDimension | str":
    """城市名 → 经纬度（CityDimension）。查不到返回错误字符串。"""
    with httpx.Client(timeout=10.0) as client:     # ← 这层包裹不能少
        def _geocode(name: str):
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json"})
            resp.raise_for_status()
            return resp.json().get("results") or []

        results = _geocode(city)
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry = _geocode(city + "市")
            if retry:
                results = retry
        if not results:
            return f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"

        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        return CityDimension(                     # ← 直接 return 维度对象
            name=loc.get("name", city),
            latitude=loc["latitude"],
            longitude=loc["longitude"],
            country=loc.get("country", ""),
            admin1=loc.get("admin1", ""),
            feature_code=str(loc.get("feature_code", "")),
            population=loc.get("population") or 0,
        )


def get_weather_from_dimension(dim: "CityDimension") -> str:
    """根据城市维度查天气。输入是 geocode_city 的产物。"""
    with httpx.Client(timeout=10.0) as client:
        try:
            weather_resp = client.get(WEATHER_URL, params={
                "latitude": dim.latitude,        # ← 用维度的经纬度，不再 geocoding
                "longitude": dim.longitude,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            weather_resp.raise_for_status()
        except httpx.RequestError as e:
            return f"天气数据获取失败：{e}"
        data = weather_resp.json()
        cur = data["current"]; daily = data["daily"]
        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
        lines = [
            f"【{dim.country} {dim.admin1} {dim.name}】天气报告",   # ← 用维度的字段拼位置
            f"坐标：{dim.latitude:.2f}°N, {dim.longitude:.2f}°E",
            "",
            f"当前天气：{weather_desc}",
            f"  温度：{cur['temperature_2m']}°C",
            f"  相对湿度：{cur['relative_humidity_2m']}%",
            f"  风速：{cur['wind_speed_10m']} km/h",
            "",
            "未来3天预报：",
        ]
        for i in range(3):                        # ← 这段格式化从老 get_weather 搬来
            day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
            lines.append(f"  {daily['time'][i]}：{day_desc}，"
                         f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
                         f"降水 {daily['precipitation_sum'][i]} mm")
        return "\n".join(lines)


def get_weather(city: str) -> str:
    """兼容入口：geocode_city + get_weather_from_dimension（三种方式旧调用不断）。"""
    dim = geocode_city(city)
    if isinstance(dim, str):          # geocode 失败时返回的是错误字符串
        return dim
    return get_weather_from_dimension(dim)


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, help="城市名称，如 宁德")
    parser.add_argument("--geocode", action="store_true",
                        help="只输出城市维度(JSON)，演示单独查询经纬度")
    args = parser.parse_args()

    dim = geocode_city(args.city)
    if isinstance(dim, str):                 # geocode 失败：透传错误字符串
        print(dim)
    elif args.geocode:                       # 姿势①：只查维度
        print(json.dumps(dim.__dict__, ensure_ascii=False, indent=2))
    else:                                    # 姿势②：维度 → 天气
        print(get_weather_from_dimension(dim))

