import google.generativeai as genai
import os
import toml

try:
    # Try to load from secrets.toml
    secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        secrets = toml.load(secrets_path)
        api_key = secrets.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            print("API Key found and configured.")
            print("Available models:")
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(f"- {m.name}")
        else:
            print("API Key not found in secrets.toml")
    else:
        print(f"secrets.toml not found at {secrets_path}")
except Exception as e:
    print(f"Error: {e}")
