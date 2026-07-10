import requests

url = "http://127.0.0.1:8000/inspect"
file_path = "c:/PROJECTS/Industrial-AI-QC/test_images/leather/Leather_1.png"

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())
