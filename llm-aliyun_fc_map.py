# Author: shen weijie
# Date: 2025-08-28
# Description: 通阿里千问大模型自然语言，调用高德地图API。
#----------------------

import json
import requests
from openai import OpenAI


# 高德地图API密钥
amap_key="09dc2b27d3e30de637b17ebb7671e412"

def get_location_coordinate(location, city="广州"):
    #用于查询某个地点的地理坐标
    url = f"https://restapi.amap.com/v5/place/text?key={amap_key}&keywords={location}&region={city}"
    print(url)
    r = requests.get(url)
    result = r.json()
    if "pois" in result and result["pois"]:
        return result["pois"][0]
    return None

def search_nearby_pois(longitude, latitude, keyword):
    #用于查询地理坐标附近的某些信息
    url = f"https://restapi.amap.com/v5/place/around?key={amap_key}&keywords={keyword}&location={longitude},{latitude}"
    print(url)
    r = requests.get(url)
    result = r.json()
    ans = ""
    if "pois" in result and result["pois"]:
        for i in range(min(3, len(result["pois"]))):
            name = result["pois"][i]["name"]
            address = result["pois"][i]["address"]
            distance = result["pois"][i]["distance"]
            ans += f"{name}\n{address}\n距离：{distance}米\n\n"
    return ans

# 定义以上二个函数get_location_coordinate及search_nearby_pois在大模型中调用的格式，用于和OpenAI的函数llm_function_calling调用接口对接
tools=[{
    "type": "function",
    "function": {
        "name": "get_location_coordinate",
        "description": "根据POI名称，获得POI的经纬度坐标",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "POI名称，必须是中文",
                },
                "city": {
                    "type": "string",
                    "description": "POI所在的城市名，必须是中文",
                }
            },
            "required": ["location", "city"],
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "search_nearby_pois",
        "description": "搜索给定坐标附近的poi",
        "parameters": {
            "type": "object",
            "properties": {
                "longitude": {
                    "type": "string",
                    "description": "中心点的经度",
                },
                "latitude": {
                    "type": "string",
                    "description": "中心点的纬度",
                },
                "keyword": {
                    "type": "string",
                    "description": "目标poi的关键字",
                }
            },
            "required": ["longitude", "latitude", "keyword"],
        }
    }
}]

    
def llm_function_calling(messages):
    """
    调用阿里百炼大模型
    """
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-d5ed06050cf44bd29d04511273dda219",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 调用大模型指定model=qwen-plus, 并传入messages, 以及函数定义tools
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools,
        temperature=0,
        max_tokens=1024
    )
    return response.choices[0].message
    
def do(prompt):
    # 定义一个问题，并调用大模型来回答

    # 定义问题
    messages = [
        {"role": "system", "content": "你是一个地图通，你可以找到任何地址。"},
        {"role": "user", "content": prompt}
    ]

    # 调用大模型，获取回复
    response = llm_function_calling(messages)
    if (response.content is None):  
        response.content = ""

    # 把大模型的回复加入到对话中
    messages.append(response)       
    #print(response)

    # 如果返回的是函数调用结果，则打印出来
    while (response.tool_calls is not None):
        #模型支持一次返回多个函数调用请求
        for tool_call in response.tool_calls:
            args = json.loads(tool_call.function.arguments)
            print(args)

            if (tool_call.function.name == "get_location_coordinate"):
                print("Call: get_location_coordinate")
                result = get_location_coordinate(**args)
            elif (tool_call.function.name == "search_nearby_pois"):
                print("Call: search_nearby_pois")
                result = search_nearby_pois(**args)

            print("=====函数返回=====")
            print(result)

            messages.append({
                "tool_call_id": tool_call.id,  # 用于标识函数调用的 ID
                "role": "tool",
                "name": tool_call.function.name,
                "content": str(result)  # 数值result 必须转成字符串
            })

        # 再次调用大模型，获取回复
        response = llm_function_calling(messages)
        if (response.content is None): 
            response.content = ""
        messages.append(response) 

    print("=====最终回复=====")
    print(response.content)

if __name__ == "__main__":
    #print(get_location_coordinate("广州天河区"))
    #print(search_nearby_pois("113.280468", "23.125177", "咖啡"))
    do("南京航空航天大学将军路附近的照相馆")

