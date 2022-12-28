import json
from Agent import Agent

agent = Agent(
    size = 15,
    win_len = 5,
    max_searches = 500
)

fullInput = json.loads(input())
requests = fullInput["requests"]
responses = fullInput["responses"]
turn = len(responses)
for i in range(turn):
    if requests[i]["x"] != -1:
        agent.update_root((requests[i]["x"], requests[i]["y"]))
    agent.update_root((responses[i]["x"], responses[i]["y"]))
if turn == 0 and requests[0]["x"] == -1:
    print(json.dumps({"response": {"x": 7, "y": 7}}))
else:
    (x, y) = agent.search((requests[turn]["x"], requests[turn]["y"]))
    print(json.dumps({"response": {"x": x, "y": y}}))