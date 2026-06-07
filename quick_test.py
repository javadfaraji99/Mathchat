"""
Quick test with a hardcoded problem
"""
from math_chat import MathChat
from utils import mylogger, load_config, setup_proxy
import os

def quick_test(config_list):
    # Hardcoded test problem
    test_problem = "Solve the quadratic equation: x^2 - 5x + 6 = 0"
    
    print(f"Testing with problem: {test_problem}")
    
    # Initialize MathChat (using test mode)
    math_chat = MathChat(
        model="test-model",
        prompt_type="v3.9python",
        prompt_location="user",
        max_round=5,
        logger=mylogger("quick_test.log"),
        use_cache=False,
        config_list=config_list
    )

    output_folder = "./test_output"
    os.makedirs(output_folder, exist_ok=True)  

    # Run test
    result = math_chat.test_single_problem(test_problem, "Algebra", file_to_be_saved=os.path.join(output_folder, "conversation.txt"))
    
    # Show answer
    print(f"\n📝 Full response:\n{result['response_with_ans']}")

if __name__ == "__main__":
    config_list = load_config("OAI_CONFIG_LIST.json")
    config_list = setup_proxy(config_list)
    quick_test(config_list)