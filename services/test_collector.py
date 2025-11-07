#!/usr/bin/env python3
import requests
import time
import json
import os

# Your VM's public IP.
# We use 127.0.0.1 (localhost) here because we're running the test
# on the VM itself, which is simpler than routing out and back in.
URL = "http://127.0.0.1:8000/state.json"

print(f"Polling live state from {URL}...")
print("Press Ctrl+C to stop.\n")

last_step = -1

try:
    while True:
        try:
            r = requests.get(URL)
            r.raise_for_status() # Raises an error if status is not 200
            
            data = r.json()
            
            # Check if the simulation has reset
            if data["step"] < last_step:
                print("--- SIMULATION RESET (RESTARTED BY SYSTEMD) ---")
            
            last_step = data["step"]
            
            # Clear the screen and print the new state
            os.system('clear')
            print(f"Successfully polled {URL}:\n")
            print(json.dumps(data, indent=2))
            
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect.")
            print("Is the 'python3 -m http.server 8000' running in ~/traffic_rl/shared?")
        except Exception as e:
            print(f"An error occurred: {e}")

        time.sleep(2) # Poll every 2 seconds

except KeyboardInterrupt:
    print("\nTest stopped.")
