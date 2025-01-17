import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

import re
from langchain_core.output_parsers import JsonOutputParser

from langchain_core.tools import tool
import requests
from langchain_core.messages import ToolMessage

from langchain.schema import AIMessage, HumanMessage, SystemMessage

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    
    match = re.search(r'(\d{4})年台灣(\d{1,2})月', question)
    if match:
        year = match.group(1)
        month = match.group(2).zfill(2) 
        #print(year)
        #print(month)
        prompt = f"列出{year}年台灣{month}月的所有紀念日，並以JSON格式呈現，每個紀念日包含日期和名稱，例如：{{'date': '年份-月份-日期', 'name': '紀念日名稱'}}。"
        response = llm.invoke(prompt)

        json_parser = JsonOutputParser()
        json_output = json_parser.invoke(response)
        result = {"Result": json_output}
    else:
        result = {"Result": []}


    print(result)
    return json.dumps(result, ensure_ascii=False, indent=2)
    #pass

@tool
def get_holidays(year: str, month: str) -> str:
    """列出指定年份台灣某月的所有紀念日."""
    url = f"https://calendarific.com/api/v2/holidays?&api_key=j18vPyhJbJojqXe39YeCXBfvJKl4Y6Mk&country=TW&year={year}&month={month}"
    response = requests.get(url)
    if response.status_code == 200:
        holidays = response.json().get('response', {}).get('holidays', [])
        formatted_holidays = [{'date': holiday['date']['iso'], 'name': holiday['name']} for holiday in holidays]
        print(type(formatted_holidays))
        result = {'Result': formatted_holidays}
        return json.dumps(result, ensure_ascii=False, indent=4)
    else:
        return {"error": "Failed to retrieve data"}
    
def generate_hw02(question):
    llm2 = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    query = question
    format_instructions = f"以JSON格式呈現，每個紀念日包含日期和名稱，例如：{{'date': '年份-月份-日期', 'name': '紀念日名稱'}}。使用繁體中文。"
    prompt = f"{query}\n\n{format_instructions}"

    # 設定工具清單
    tools = [get_holidays]


    # 設定 LLM 並綁定工具
    llm_with_tools = llm2.bind_tools(tools)
    messages = [HumanMessage(prompt)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    for tool_call in ai_msg.tool_calls:
        selected_tool = {"get_holidays": get_holidays}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    response = llm_with_tools.invoke(messages)
    # print(response.content)

    # 使用正則表達式提取JSON數據
    json_match = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
    if json_match:
        json_string = json_match.group(1)
    else:
        raise ValueError("未能提取到 JSON 字符串")

    # 解析JSON數據
    holidays = json.loads(json_string)

    # 構建所需的JSON結構
    result = {"Result": holidays}

    # 轉換為格式化的JSON字符串
    formatted_result_json = json.dumps(result, ensure_ascii=False, indent=4)

    # 輸出結果
    return formatted_result_json
    #pass
    
def generate_hw03(question2, question3):
    llm3 = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    NeedToAdd = False
    chat_history = []  # Use a list to store messages
    # Set an initial system message (optional)
    system_message = SystemMessage(content="You are a helpful AI assistant.")
    chat_history.append(system_message)  # Add system message to chat history
    chat_history.append(HumanMessage(content=question2))
    tools = [get_holidays]
    llm_with_tools_3 = llm3.bind_tools(tools)
    ai_msg = llm_with_tools_3.invoke(chat_history)
    chat_history.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"get_holidays": get_holidays}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        chat_history.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    result = llm_with_tools_3.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    
    chat_history.append(HumanMessage(content=question3))  # Add user message
    # Get AI response using history
    result = llm3.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))  # Add AI message
    reason = response

    if "沒" in response and "清單" in response:
        NeedToAdd = True  # 需要加入到清單
        query = f"如果節日不在該月份清單內的話，把它加入。"
    else:
        NeedToAdd = False
    chat_history.append(HumanMessage(content=query))  # Add user message
    # Get AI response using history
    result = llm3.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))  # Add AI message
    d = {"add": NeedToAdd, "reason":reason}
    final_result = {"Result": d}
    return json.dumps(final_result, ensure_ascii=False, indent=4)

    #pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response
