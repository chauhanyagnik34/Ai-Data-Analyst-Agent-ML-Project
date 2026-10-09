"""
llm_client.py
-------------
Modular API client interface for connecting to instruction-tuned LLMs.
Uses the modern Google GenAI SDK (google-genai).

Features:
- Secure API key retrieval via python-dotenv (.env)
- Status check function (verifies key presence before network calls)
- JSON Mode support for structured query planning
- Graceful handling of network timeouts, quota limits, and missing keys
- Modular architecture: easy to swap with OpenAI, Groq, or Anthropic
"""

import os
import json
from dotenv import load_dotenv

# Load variables from .env file into os.environ
load_dotenv()

# Default model configuration
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def get_api_key() -> str:
    """Retrieves the Gemini API key from environment variables."""
    return os.getenv("GEMINI_API_KEY", "").strip()


def check_api_status() -> dict:
    """
    Checks if the LLM API key is properly configured.

    Returns:
        dict: {
            "configured": bool,
            "provider": "Google Gemini",
            "model": str,
            "message": str
        }
    """
    api_key = get_api_key()
    placeholder_keys = ["your_gemini_api_key_here", "your_key_here", ""]

    if not api_key or api_key in placeholder_keys:
        return {
            "configured": False,
            "provider": "Google Gemini",
            "model": DEFAULT_MODEL,
            "message": (
                "API key not found. Please obtain a free Gemini API key at "
                "https://aistudio.google.com/app/apikey and add it to your .env file."
            ),
        }

    return {
        "configured": True,
        "provider": "Google Gemini",
        "model": DEFAULT_MODEL,
        "message": f"API key configured successfully for model '{DEFAULT_MODEL}'.",
    }


def call_llm(
    prompt: str,
    system_instruction: str = None,
    json_mode: bool = False,
    model_name: str = None,
) -> tuple:
    """
    Calls the LLM with a prompt and returns the text response safely.

    Parameters:
        prompt (str): The user/task prompt.
        system_instruction (str, optional): System level instruction or persona.
        json_mode (bool): When True, instructs model to return strict JSON.
        model_name (str, optional): Target model name (defaults to GEMINI_MODEL).

    Returns:
        tuple: (response_text, error_message)
               - response_text: str if successful, else None
               - error_message: str describing error, or None if successful
    """
    status = check_api_status()
    if not status["configured"]:
        return None, status["message"]

    target_model = model_name or DEFAULT_MODEL
    api_key = get_api_key()

    try:
        from google import genai
        from google.genai import types

        # Initialize the official GenAI client
        client = genai.Client(api_key=api_key)

        # Configure request options
        config_kwargs = {"temperature": 0.1}

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_kwargs)

        # Send request to Gemini API
        response = client.models.generate_content(
            model=target_model,
            contents=prompt,
            config=config,
        )

        if not response or not response.text:
            return None, "Error: The model returned an empty response."

        output_text = response.text.strip()

        # If JSON mode requested, clean possible markdown wrappers
        if json_mode:
            output_text = _clean_json_markdown(output_text)

        return output_text, None

    except Exception as e:
        err_str = str(e)
        if "API_KEY_INVALID" in err_str or "400" in err_str:
            return None, "Error: Invalid Gemini API key. Please check the key in your .env file."
        elif "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
            return None, "Error: Gemini API rate limit or quota exceeded. Please wait a moment and try again."
        return None, f"LLM API Error: {err_str}"


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
    print("CHECKING LLM CLIENT CONFIGURATION")
    print("=" * 60)

    status = check_api_status()
    print(f"Provider:    {status['provider']}")
    print(f"Model:       {status['model']}")
    print(f"Configured:  {status['configured']}")
    print(f"Message:     {status['message']}")

    if status["configured"]:
        print("\nSending a quick test ping to Gemini API...")
        res, err = call_llm("Return only the word: PONG", json_mode=False)
        if err:
            print(f"Ping failed: {err}")
        else:
            print(f"LLM Response: {res}")
            print("SUCCESS: LLM client is live and operational!")
    else:
        print("\nNote: Add your free Gemini API key to .env whenever you're ready to test live calls.")
