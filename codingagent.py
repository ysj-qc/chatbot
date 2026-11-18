import os
import json
import subprocess
from openai import OpenAI

API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")

if not API_KEY or not BASE_URL:
    raise RuntimeError("Please set API_KEY & BASE_URL first before run this script!")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 用字典格式描述模型可以使用的工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读指定文件的内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件名"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入指定文件，会覆盖已有内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "执行一条shell命令并返回结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"}
                },
                "required": ["command"]
            }
        }
    }
]

# 工具实现
def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：文件 {path} 不存在"

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {path}"

def run_command(command):
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, errors="replace", timeout=10
        )
        output = result.stdout
        # shell 惯例：返回码 0 表示成功，非 0 表示出错
        if result.returncode != 0:
            output += f"\n[错误] {result.stderr}"
        return output or "(无输出)"
    except subprocess.TimeoutExpired:
        return "[错误] 命令执行超时（10秒）"

tool_functions = {
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}

SYSTEM_PROMPT = """你是一个编程助手。你可以读写文件和执行命令来帮用户完成编程任务。
工作流程：先理解需求，写代码，然后运行验证。如果发现错误就修复并重新运行，直到确认正确。"""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("编程助手已激活，请输入任务以开始，输入q以退出\n")

while True:
    user_input = input("你：")
    if user_input.strip() == "q":
        break

    messages.append({"role": "user", "content": user_input})

    #每一次问答内的Agent循环，模型思考并行动，直到自己觉得可以停下来
    while True:
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=messages,
            tools=tools,
            # 开启思考模式，让模型每一步决策前先推理
            extra_body={"reasoning_effort": "high"}
        )
        assistant_message = response.choices[0].message

        #不管这轮是否调用工具，assistant消息先加到上下文
        messages.append(assistant_message)

        #assistant消息可以同时带文本和工具调用，显示一下模型本轮说的话
        if assistant_message.content:
            print(f"AI: {assistant_message.content}\n")

        #没工具调用就说明完成任务，可以退出问答循环了
        if not assistant_message.tool_calls:
            break
        
        #多个工具的请求需要依次执行
        for tool_call in assistant_message.tool_calls:
            # arguments 是 JSON 字符串，需要解析成字典
            # **args 把字典解包成关键字参数
            # tool_calling的三步
            args = json.loads(tool_call.function.arguments)
            func = tool_functions[tool_call.function.name]
            result = func(**args)
            #打印工具调用过程，观察Agent行为
            print(f"  [工具] {tool_call.function.name}({args})")
            #只打印前200个字符，限制大文件
            print(f"  [结果] {result[:200]}")

            # 工具调用结果，追加到上下文
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })