from utility import install_packages, define_custom_llm, create_llm_predictor, load_documents, create_prompt_helper, create_gpt_list_index, query_index
from helper import check_directory_exists, check_and_install_packages

def main():
    check_and_install_packages()
    data_directory = '/content/data'
    if not check_directory_exists(data_directory):
        print(f"Directory {data_directory} does not exist.")
        return
    FlanLLM = define_custom_llm()
    llm_predictor = create_llm_predictor(FlanLLM)
    hfemb = HuggingFaceEmbeddings()
    embed_model = LangchainEmbedding(hfemb)
    documents = load_documents(data_directory)
    num_output = 512
    max_input_size = 512
    max_chunk_overlap = 20
    prompt_helper = create_prompt_helper(max_input_size, num_output, max_chunk_overlap)
    index = create_gpt_list_index(documents, embed_model, llm_predictor, prompt_helper)
    query_index(index, "")

if __name__ == "__main__":
    main()
