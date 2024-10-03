import json
import re
import time

import requests
from tqdm import tqdm

class BharatMetricEvaluator:
    def __init__(self):
        self.client = "http://localhost:11434/api/chat"  # Replace with your local IP and port if needed

    def read_json_file(self, file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def extract_answer(response):
        if response.strip():  # Checks if the response is not empty and not just whitespace
            match = re.search(r"ANSWER:?\s*([A-D])", response, re.IGNORECASE)
            if match:
                return match.group(1).upper()  # Return the matched letter in uppercase
        return None

    def ask_llm(self, question, answers):
        options = ', '.join([f"{key}) {value}" for key, value in answers.items()])
        prompt = f"Question: {question}\nOptions: {options}\n\nChoose the correct answer (A, B, C, or D) only. Always return in this format: 'ANSWER: X' "
        response = self.send_request(prompt)
        if response:
            result = self.extract_answer(response)
            return result

    def send_request(self, prompt, false=None):
        try:
            data = {
                "model": "llama3",
                "messages": [
                    {"role": "system", "content": "You are a security expert who answers questions."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False
            }
            response = requests.post(self.client, data=json.dumps(data), headers={'Content-Type': 'application/json'})

            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.text
        except Exception as e:
            print(f"Error: {e}. Attempting the question again.")
            time.sleep(2)
            return self.send_request(prompt)

    def run_evaluation(self, file_path):
        json_data = self.read_json_file(file_path)
        questions_data = json_data['questions']

        correct_count = 0
        incorrect_answers = []

        with tqdm(total=len(questions_data), desc="Processing Questions") as progress_bar:
            for item in questions_data:
                question = item['question']
                answers = item['answers']
                correct_answer = item['solution']

                llm_answer = self.ask_llm(question, answers)
                if llm_answer == correct_answer:
                    correct_count += 1
                else:
                    incorrect_answers.append({
                        'question': question,
                        'correct_answer': correct_answer,
                        'llm_answer': llm_answer
                    })

                accuracy_rate = correct_count / (progress_bar.n + 1) * 100
                progress_bar.set_postfix_str(f"Accuracy: {accuracy_rate:.2f}%")
                progress_bar.update(1)

        print(f"Final Accuracy: {correct_count / len(questions_data) * 100}%")

        if incorrect_answers:
            print("\nIncorrect Answers:")
            for item in incorrect_answers:
                print(f"Question: {item['question']}")
                print(f"Expected Answer: {item['correct_answer']}, LLM Answer: {item['llm_answer']}\n")

# Example usage:
if __name__ == "__main__":
    file_path='CyberMetric-1-v1.json'
    evaluator = BharatMetricEvaluator()
    evaluator.run_evaluation(file_path)