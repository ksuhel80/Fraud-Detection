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
        "-t", "fraud-detection-api", 
        "api/"
    ], check=True)
    
    # Run Docker container
    subprocess.run([
        "docker", "run", 
        "-d", 
        "-p", f"{config['api']['port']}:{config['api']['port']}",
        "--name", "fraud-detection-api",
        "fraud-detection-api"
    ], check=True)
    
    print("API deployed successfully!")

if __name__ == "__main__":
    main()