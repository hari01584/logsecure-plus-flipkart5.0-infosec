# import connector.base
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import torch
import transformers
from langchain import HuggingFacePipeline
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain import HuggingFacePipeline
from langchain import LLMChain
from langchain.llms import OpenAI

TYPE = ["falcon", "openai"]

class Connector():
    def __init__(self, type=TYPE[0]):
        if type == TYPE[0]:
            self.connector = FalconConnector()
        else:
            self.connector = GPT3Connector()

    def connect(self):
        return self.connector.connect()

    def disconnect(self):
        return self.connector.disconnect()

    def evaluate(self, query):
        return self.connector.evaluate(query)
    
    # -- Internal function to craft a prompt! -- #
    def _craft_prompt(self, query):
        rules = query["rules"]
        data = query["data"]
        format = query["format"]
        understand_data = query["understand_data"]
        sample_examples = query["sample_examples"]

        template = "Given set of compliances, check if the data is compliant, if it is not then show remediation steps.\n" 

        template += f"compliances are: {rules}\n"
        template += f"data is: {data}\n"

        # Add extra fields if present
        if (understand_data):
            template += f"information required to understand data: {understand_data}\n"

        if (sample_examples):
            template += f"example case :{sample_examples}\n"
        
        generated_prompt = template

        print("Generated prompt is: ", generated_prompt)

        return generated_prompt


class FalconConnector(Connector):
    # Configure our model path
    MODEL_PATH = "models/llm/7b_falcon_sharded"

    def __init__(self):
        # Logic to initilize logic and model for this!
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_id = self.MODEL_PATH
        model_4bit = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(model_4bit)

        # Create pipeline

        self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_4bit,
                tokenizer=tokenizer,
                use_cache=True,
                device_map="1000",
                max_length=296,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
        )

        print("Creating final llm pipeline now!")
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)

        print("Init prompt template")

    def connect(self):
        print("Already connected")

    def disconnect(self):
        pass

    def evaluate(self, query):
        return self.llm(self._craft_prompt(query))

if __name__ == "__main__":
    # A unit test for the falcon connector
    connector = Connector()
    connector.connect()
    print(connector.evaluate({
        "rules": "name shouldn't contain abc",
        "data": "name: abc",
        "format": "name: string"
    }))
    connector.disconnect()

class GPT3Connector(Connector):
    MYGPTKEY = ""

    def __init__(self):
        self.llm = OpenAI(model_name="text-davinci-003")

    def connect(self):
        pass

    def disconnect(self):
        pass

    def evaluate(self, query):
        return self.llm(self._craft_prompt(query))