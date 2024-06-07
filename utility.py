from transformers import pipeline
from langchain.llms.base import LLM
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, PromptHelper
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LLMPredictor
import torch

def define_custom_llm():
    class FlanLLM(LLM):
        model_name = "google/flan-t5-base"
        pipeline = pipeline("text2text-generation", model=model_name, device=0, model_kwargs={"torch_dtype":torch.bfloat16})

        def _call(self, prompt, stop=None):
            return self.pipeline(prompt, max_length=9999)[0]["generated_text"]

        def _identifying_params(self):
            return {"name_of_model": self.model_name}

        def _llm_type(self):
            return "custom"
    
    return FlanLLM()

def create_llm_predictor(llm_class):
    return LLMPredictor(llm=llm_class)

def load_documents(directory):
    return SimpleDirectoryReader(directory).load_data()

def create_prompt_helper(max_input_size, num_output, max_chunk_overlap):
    return PromptHelper(max_input_size, num_output, max_chunk_overlap)

def create_gpt_list_index(documents, embed_model, llm_predictor, prompt_helper):
    return GPTListIndex(documents, embed_model=embed_model, llm_predictor=llm_predictor)

def query_index(index, query_text):
    response = index.query(query_text)
    print(response)
