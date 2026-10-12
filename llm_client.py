"""
llm_client.py
-------------
Modular API client interface for connecting to local Ollama instruction-tuned LLMs.
Uses the standard local Ollama HTTP API (http://127.0.0.1:11434).

Features:
- Completely offline, privacy-first, zero-cost local LLM execution
- Configurable base URL and model via python-dotenv (.env)
- JSON Mode support via Ollama's native structured format: "json"
- Fast health checks and model availability detection
- Graceful error handling for offline or unreachable server
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load variables from .env file into os.environ
load_dotenv()

# Default local Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def get_base_url() -> str:
    """Retrieves the Ollama base endpoint URL from environment variables."""
    return os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL).rstrip("/")


def get_default_model() -> str:
    """Retrieves the default Ollama model name from environment variables."""
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL).strip()


def check_api_status() -> dict:
    """
    Checks if the local Ollama server is running and the target model is available.

    Returns:
        dict: {
            "configured": bool,
            "provider": "Ollama (Local)",
            "model": str,
            "message": str
        }
    """
    base_url = get_base_url()
    target_model = get_default_model()

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Match model name with or without tag (e.g. 'llama3.2' or 'llama3.2:latest')
            matched = any(
                m == target_model or m.startswith(f"{target_model}:")
                for m in models
            )

            if matched:
                return {
                    "configured": True,
                    "provider": "Ollama (Local)",
                    "model": target_model,
                    "message": f"Local Ollama is online with model '{target_model}'.",
                }
            else:
                available_str = ", ".join(models) if models else "None"
                return {
                    "configured": False,
                    "provider": "Ollama (Local)",
                    "model": target_model,
                    "message": (
                        f"Ollama is running, but model '{target_model}' was not found. "
                        f"Installed models: [{available_str}]. Run 'ollama pull {target_model}' in your terminal."
                    ),
                }
        else:
            return {
                "configured": False,
                "provider": "Ollama (Local)",
                "model": target_model,
                "message": f"Ollama server returned HTTP {response.status_code}.",
            }

    except requests.exceptions.RequestException as e:
        return {
            "configured": False,
            "provider": "Ollama (Local)",
            "model": target_model,
            "message": (
                f"Could not connect to local Ollama at {base_url}. "
                "Ensure Ollama is running by executing 'ollama serve' in your terminal."
            ),
        }


def call_llm(
    prompt: str,
    system_instruction: str = None,
    json_mode: bool = False,
    model_name: str = None,
    timeout: int = 60,
) -> tuple:
    """
    Calls the local Ollama LLM with a prompt and returns the text response safely.

    Parameters:
        prompt (str): The user/task prompt.
        system_instruction (str, optional): System level instruction or persona.
        json_mode (bool): When True, instructs model to return strict JSON.
        model_name (str, optional): Target model name (defaults to OLLAMA_MODEL).
        timeout (int): Timeout in seconds for the generation request.

    Returns:
        tuple: (response_text, error_message)
               - response_text: str if successful, else None
               - error_message: str describing error, or None if successful
    """
    status = check_api_status()
    if not status["configured"]:
        return None, status["message"]

    target_model = model_name or get_default_model()
    base_url = get_base_url()

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": target_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    if json_mode:
        payload["format"] = "json"

    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=timeout,
        )

        if response.status_code != 200:
            return None, f"Ollama HTTP {response.status_code}: {response.text}"

        data = response.json()
        message = data.get("message", {})
        output_text = message.get("content", "").strip()

        if not output_text:
            return None, "Error: The model returned an empty response."

        if json_mode:
            output_text = _clean_json_markdown(output_text)

        return output_text, None

    except requests.exceptions.Timeout:
        return None, "Error: Ollama request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"Ollama Connection Error: {str(e)}"
    except Exception as e:
        return None, f"LLM Error: {str(e)}"


def _clean_json_markdown(text: str) -> str:
    """Strips markdown ```json ``` wrappers if returned by the LLM."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


# Standalone demonstration block
if __name__ == "__main__":
    print("=" * 60)
    print("CHECKING LOCAL OLLAMA CLIENT CONFIGURATION")
    print("=" * 60)

    status = check_api_status()
    print(f"Provider:    {status['provider']}")
    print(f"Model:       {status['model']}")
    print(f"Configured:  {status['configured']}")
    print(f"Message:     {status['message']}")

    if status["configured"]:
        print("\nSending a quick test ping to local Ollama Llama 3.2...")
        res, err = call_llm("Return only the word: PONG", json_mode=False)
        if err:
            print(f"Ping failed: {err}")
        else:
            print(f"LLM Response: {res}")
            print("SUCCESS: Ollama Llama 3.2 is live and operational!")
    else:
        print("\nNote: Ensure Ollama is running and model llama3.2 is pulled.")
