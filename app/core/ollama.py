import requests
import json

def query_llama(prompt):
    # Replace the URL with the endpoint for your local Llama 3.2 API
    url = "http://localhost:11434/api/generate/"  
    payload = {
        "prompt": prompt,
        "max_tokens": 150,      # Adjust as needed
        "temperature": 0.7      # Adjust as needed
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raises an exception for HTTP errors
    except requests.RequestException as e:
        return f"Error during request: {e}"
    
    # Assuming the API returns JSON with a key 'response' or 'text'
    data = response.json()
    return data.get("response") or data.get("text") or "No valid response key found."

if __name__ == "__main__":
    query_input = input("Enter your query: ")
    result = query_llama(query_input)
    print("Llama 3.2 Response:")
    print(result)
