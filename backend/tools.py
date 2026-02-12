import time

def reminder(day: tuple, hour: tuple, minute: tuple, remind: str):
    if day[1]:
        remind_day = int(time.strftime('%d')) + day[0]
    if hour[1]:
        remind_day = int(time.strftime('%d')) + day[0]
    if minute[1]:
        remind_day = int(time.strftime('%d')) + day[0]
    with open("data/reminders.txt", "a", encoding="utf-8") as f:
        f.write(f"{remind_day}, {hour}: {minute}")
        
def tool_calls_history(response):
    if response.choice[0].message.tool_calls:
        for i in range(0, len(response.choice[0].message.tool_calls)):
            print(f"Tool calls {i+1}:")
            print(f"Function: {response.choice[0].message.tool_calls[i].function.name}")
            print(f"Arguments: {response.choice[0].message.tool_calls[i].function.argument}")
            