# import subprocess
# import itertools
# import os
# import re
# import json

# # 1. Define the grid of values you want to test
# skews = [0.02, 0.05, 0.1, 0.2, 0.4]

# results = []

# ladders_to_test = [
#     [20, 0, 0],    # Aggressive (All at the front)
#     [5, 10, 5],    # Balanced (Bell curve)
#     [2, 3, 15],    # Passive (Wait for the big spikes)
#     [10, 5, 5]     # Front-loaded
# ]

# print(f"{'Skew':<6} | {'Spread':<8} | {'Profit':<10}")
# print("-" * 30)

# # 2. Loop through every combination
# for skew, ladder in itertools.product(skews, ladders_to_test):
#     # Prepare the environment with our test parameters
#     current_env = os.environ.copy()
#     current_env["SKEW"] = str(skew)
#     current_env["LADDER"] = json.dumps(ladder)

#     # 3. Run the backtester command
#     cmd = ["uv", "run", "prosperity4btest", "./trader.py", "0"]
#     process = subprocess.run(cmd, env=current_env, capture_output=True, text=True)
    
#     # 4. Extract the profit from the backtester output
#     # This regex looks for a number after "Total profit:" in the logs
#     match = re.search(r"sharpe_ratio:\s+([\d.-]+)", process.stdout)
#     profit = float(match.group(1)) if match else 0.0
    
#     results.append((skew, ladder, profit))
#     print(f"{skew:<6} | {str(ladder):<8} | {profit:<10.2f}")

# # 5. Find the winner
# best = max(results, key=lambda x: x[2])
# print(f"\n🏆 BEST SETTING: Skew {best[0]}, Ladder {best[1]} with Profit {best[2]}")