import json
import openai


def load_config(config_path="OAI_CONFIG_LIST.json"):
    """Load credentials from JSON config file."""
    with open(config_path, "r") as f:
        config_list = json.load(f)
    return config_list


def main():
    # -----------------------------------------
    # Load credentials from JSON
    # -----------------------------------------
    config_list = load_config()
    config = config_list[0]  # Use the first config entry

    # -----------------------------------------
    # Set openai module-level settings (v0.27.x style)
    # -----------------------------------------
    openai.api_key = config["api_key"]
    openai.api_base = config["api_base"]

    model = config["model"]

    # print(f"OpenAI version : {openai.__version__}")
    print(f"Using model    : {model}")
    print(f"Using base     : {openai.api_base}")
    print("-" * 50)

    # -----------------------------------------
    # Send a test chat completion request
    # -----------------------------------------
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Explain what an API is in 3 sentences."
            },
        ],
        temperature=0.7,
        max_tokens=256,
    )

    # -----------------------------------------
    # Print the result
    # -----------------------------------------
    answer = response["choices"][0]["message"]["content"]
    usage = response["usage"]

    print("Response:\n")
    print(answer)
    print("-" * 50)
    print(f"Tokens used — prompt: {usage['prompt_tokens']}, "
          f"completion: {usage['completion_tokens']}, "
          f"total: {usage['total_tokens']}")


if __name__ == "__main__":
    main()