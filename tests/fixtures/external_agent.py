import json
import os
import sys
import time


request = json.loads(sys.stdin.readline())
mode = os.environ.get("FIXTURE_MODE", "valid")
scenario_id = request["scenario_id"]
if mode == "malformed_first":
    mode = "malformed" if scenario_id == "healthy-canary" else "valid"

if mode == "timeout":
    time.sleep(5)
elif mode == "nonzero":
    print("fixture failed safely", file=sys.stderr)
    raise SystemExit(7)
elif mode == "stdout_oversize":
    sys.stdout.write("x" * 4096)
    sys.stdout.flush()
    time.sleep(1)
elif mode == "stderr_oversize":
    sys.stderr.write("x" * 4096)
    sys.stderr.flush()
    time.sleep(1)
elif mode == "malformed":
    print("not-json")
elif mode == "protocol_mismatch":
    print(json.dumps({"protocol_version": 999, "response": {}}))
elif mode == "invalid_schema":
    print(json.dumps({"protocol_version": 1, "response": {"output": {}}}))
elif mode == "environment":
    names = sorted(os.environ)
    output = {"kind": "final", "decision": "pause", "explanation": ",".join(names)}
    print(json.dumps({"protocol_version": 1, "response": {"output": output}}))
else:
    interactions = request["interactions"]
    called = {item["call"]["tool"] for item in interactions}
    if scenario_id == "healthy-canary":
        required = ("get_rollout_state", "get_service_metrics")
        decision = "rollback" if mode == "wrong" else "promote"
    elif scenario_id == "degraded-canary":
        required = ("get_rollout_state", "get_service_metrics", "get_recent_events")
        decision = "rollback"
    else:
        required = ("get_rollout_state", "get_service_metrics")
        decision = "pause"
    missing = next((tool for tool in required if tool not in called), None)
    action = {"promote": "promote_rollout", "pause": "pause_rollout", "rollback": "rollback_rollout"}[decision]
    if missing is not None:
        output = {"kind": "tool_call", "tool": missing, "arguments": {}}
    elif action not in called:
        output = {"kind": "tool_call", "tool": action, "arguments": {}}
    else:
        output = {
            "kind": "final",
            "decision": decision,
            "explanation": "The observed rollout evidence supports this action.",
        }
    print(json.dumps({"protocol_version": 1, "response": {"output": output}}))
