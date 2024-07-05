from huggingface_hub import InferenceClient
import json
import yaml

class ChatDocumentQA:
    def __init__(self) -> None:
      hf_key = "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"
      self.mistral_client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1",token=hf_key)

    def format_prompt(self, question: str, data: str) -> str:
        """
        Formats the prompt for the language model.
        Args:
            question (str): The user's question.
            data (str): The data to be analyzed.
        Returns:
            str: Formatted prompt.
        """
        prompt = "<s>"
        prompt = f"""[INST] you are the german language and universal language expert .your task is  analyze the given data and user ask any question about given data answer to the user question.your returning answer must in user's language.otherwise reply i don't know.
            data:{data}
            question:{question}[/INST]"""

        prompt1 = f"[INST] {question} [/INST]"
        return prompt+prompt1
    
    def yaml_file_to_json(self,yaml_file):
        with open(yaml_file, 'r') as f:
            # Load YAML content from file
            yaml_data = yaml.safe_load(f)
            # Convert YAML to JSON
            json_data = json.dumps(yaml_data, indent=2)
        return json_data
    def generate(self, question: str,data_path) -> str:
        """
        Generates text based on the prompt and transcribed text.
        Args:
            prompt (str): The prompt for generating text.
            transcribed_text (str): The transcribed text for analysis.
            temperature (float): Controls the randomness of the sampling. Default is 0.9.
            max_new_tokens (int): Maximum number of tokens to generate. Default is 5000.
            top_p (float): Nucleus sampling parameter. Default is 0.95.
            repetition_penalty (float): Penalty for repeating the same token. Default is 1.0.
        Returns:
            str: Generated text.
        """
        # try:
        temperature=0.9
        max_new_tokens=1000
        top_p=0.95
        repetition_penalty=1.0  

        temperature = float(temperature)
        if temperature < 1e-2:
            temperature = 1e-2
        top_p = float(top_p)

        generate_kwargs = dict(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            seed=42,
        )

        # Example usage:
        yaml_file = data_path  # Replace 'example.yaml' with your YAML file path
        data = self.yaml_file_to_json(yaml_file)

        prompt = self.format_prompt(question, data)
        # Generate text using the mistral client
        stream = self.mistral_client.text_generation(prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
        output = ""
        # Concatenate generated text
        for response in stream:
            output += response.token.text
        return output.replace("</s>","")

    def main(self,question):
        
        data_path = "D:\\python_pycham1\\domain.yml"
        result = self.generate(question,data_path)
        print(result)
        return result


if __name__ == "__main__":
    chatdocumentqa = ChatDocumentQA()
    question = ""
    q = True
    while q:
        question = input("Question:")
        if question == "q":
            q = False
        else:
            chatdocumentqa.main(question)

