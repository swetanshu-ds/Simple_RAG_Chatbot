import requests

def collect_complaint():
    complaint_data = {}
    print("Bot: I'm sorry to hear that. Let's collect your complaint.")

    for field in ["name", "phone_number", "email", "complaint_details"]:
        complaint_data[field] = input(f"Bot: Please enter your {field.replace('_', ' ')}:\nUser: ")

    response = requests.post("http://127.0.0.1:8000/complaints", json=complaint_data)
    if response.status_code == 200:
        complaint_id = response.json().get("complaint_id")
        print(f"Bot: Your complaint has been registered with ID: {complaint_id}")
    else:
        print("Bot: Failed to create complaint. Reason:", response.text)

if __name__ == "__main__":
    collect_complaint()