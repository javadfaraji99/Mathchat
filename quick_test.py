"""
Quick test with a hardcoded problem
"""
from math_chat import MathChat
from utils import mylogger

def quick_test():
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
        use_cache=False
    )
    
    # Run test
    result = math_chat.test_single_problem(test_problem, "Algebra")
    
    # Show answer
    if "\\boxed{" in result['response_with_ans']:
        response = result['response_with_ans']
        start = response.find("\\boxed{") + 7
        end = response.find("}", start)
        print(f"\n🎯 Answer: {response[start:end]}")
    else:
        print(f"\nResponse: {result['response_with_ans']}")

if __name__ == "__main__":
    quick_test()