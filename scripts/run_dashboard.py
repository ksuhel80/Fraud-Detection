import os
import subprocess
import yaml

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Build Docker image
    subprocess.run([
        "docker", "build", 
        "-t", "fraud-detection-dashboard", 
        "dashboard/"
    ], check=True)
    
    # Run Docker container
    subprocess.run([
        "docker", "run", 
        "-d", 
        "-p", f"{config['dashboard']['port']}:{config['dashboard']['port']}",
        "--name", "fraud-detection-dashboard",
        "fraud-detection-dashboard"
    ], check=True)
    
    print("Dashboard deployed successfully!")

if __name__ == "__main__":
    main()