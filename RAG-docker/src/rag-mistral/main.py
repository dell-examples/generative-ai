### Dell Proof of Concept RAG chatbot
### Not for production use, for educational purposes only

## THESE VARIABLES MUST APPEAR BEFORE TORCH OR CUDA IS IMPORTED
## set visible GPU devices and order of IDs to the PCI bus order
## target the L40s that is on ID 1

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"   

## this integer corresponds to the ID of the GPU, for multiple GPU use "0,1,2,3"...
## to disable all GPUs, simply put empty quotes ""

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


hf_token = os.environ.get("HF_TOKEN")
print("HuggingFace Token:", hf_token)

from huggingface_hub import login
login(token=hf_token)

from langchain import HuggingFacePipeline, PromptTemplate

### import loaders
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

### for embedding
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter

#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

### for langchain chain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
#from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, pipeline, TextIteratorStreamer, AutoModelForCausalLM
from langchain.chains import LLMChain

## for quantization
from transformers import BitsAndBytesConfig


### status bars and UI and other accessories
from tqdm import tqdm
import time
import gradio as gr
import json
import torch
import sys
import gc

#################

## Constants


MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
# MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))


memory = ConversationBufferWindowMemory(
    k=5, ## number of interactions to keep in memory
    memory_key="chat_history",
    return_messages=True,  ## formats the chat_history into HumanMessage and AImessage entity list
    input_key="query",   ### for straight retrievalQA chain
    output_key="result"   ### for straight retrievalQA chain

)



#################

def info():
    print("___________Info___________")
    print("_____Device Environment settings____")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("CUDA_DEVICE_ORDER:", os.environ.get("CUDA_DEVICE_ORDER"))
    print("_____Python, Pytorch, Cuda info____")
    print("__Python VERSION:", sys.version)
    print("__pyTorch VERSION:", torch.__version__)
    print("__CUDA RUNTIME API VERSION")
    print("__CUDNN VERSION:", torch.backends.cudnn.version())
    print("_____Device assignments____")
    print("Number CUDA Devices:", torch.cuda.device_count())
    print("Current cuda device: ", torch.cuda.current_device())
    print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))


#################


def load_docs():

    global vectordb
    global embeddings

    pdf_dir_loader = PyPDFDirectoryLoader("samples/pdf-files/")
    events_loader = CSVLoader("samples/csv-files/dtw24-concierge-events-04-22-24-csv.csv", encoding='windows-1252')
#    general_info_loader = CSVLoader("samples/csv-files/dtw24-concierge-QA-dataset-04-24-2024.csv", encoding='windows-1252')
#    ppt_loader = UnstructuredPowerPointLoader("ppt-content/pan-dell-generative-ai-presentation.pptx")

    from langchain_community.document_loaders.merge import MergedDataLoader

    loader_all = MergedDataLoader(loaders=[events_loader, pdf_dir_loader])

    docs = loader_all.load()
    len(docs)


    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
    )


    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    len(texts)

    time
    vectordb = Chroma.from_documents(texts, embeddings, persist_directory="vector-db")
    print('\n' + 'Time to complete:')


#################


def prepare_model():
    # Prepare Chat model
    
    global model_id
    global model
    global tokenizer

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    print(f"Loading {model_id}")
    

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
    #    load_in_4bit=True,
    #    torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",

    )


    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.use_default_system_prompt = False
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"



################# get list of all file URLs in vector db


def get_unique_files():
    
    db = vectordb
    print("\nEmbedding keys:", db.get().keys())
    print("\nNumber of embedded docs:", len(db.get()["ids"]))
    
    # Print the list of source files
    # for x in range(len(db.get()["ids"])):
    #     # print(db.get()["metadatas"][x])
    #     doc = db.get()["metadatas"][x]
    #     source = doc["source"]
    #     print(source)
    
    # db.get()
    
    file_list = []
    
    for x in range(len(db.get()["ids"])):
        doc = db.get()["metadatas"][x]
        source = doc["source"]
        # print(source)
        file_list.append(source)
        
    ### Set only stores a value once even if it is inserted more than once.
    list_set = set(file_list)
    unique_list = (list(list_set))

    print("\nList of unique files in db:\n")
    for unique_file in unique_list:
        print(unique_file)

    pretty_files = json.dumps(unique_list, indent=4, default=str)

    print(pretty_files)

    return pretty_files

#################


def get_sources():

    res_dict = {
        "answer_from_llm": response["result"],   ### looks up result key from raw output
    }

    res_dict["source_documents"] = []    ### create an empty array for source documents key front result dict

    for each_source in response["source_documents"]:
        res_dict["source_documents"].append({
            "page_content": each_source.page_content,
            "metadata":  each_source.metadata
        })

    # print(res_dict["answer_from_llm"])  ### PRINT JUST THE RAW ANSWER FROM LLM

    pretty_sources = json.dumps(res_dict["source_documents"], indent=4, default=str)

    print(pretty_sources)

    return pretty_sources

#################


def get_model_info ():

    model_details = (
    
    f"\nGeneral Model Info:\n"
    f"\n-------------------\n"
    
    f"\n Model_id: {model_id} \n"
    f"\n Model config: {model} \n"

    f"\nGeneral Embeddings Info:\n"
    f"\n-------------------\n"

    f"\n Embeddings model config: {embeddings} \n" 

    )
        
    return model_details

#################


def process_input(
    question,
    chat_history,
    rag_toggle,
    system_prompt,
    source_docs_qty,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    repetition_penalty
                 ):


    ### system prompt variable is typed in by the user in Gradio advanced settings text box and sent into process_input function
    ### This is Llama2 prompt format 
    ### https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    
    global response
    
    
    prompt_template_rag = "\n\n [INST] <<SYS>>" + system_prompt + "<</SYS>>\n\n Context: {context} \n\n  Question: {question} \n\n[/INST]".strip()


    PROMPT_rag = PromptTemplate(template=prompt_template_rag, input_variables=["context", "question"])


    prompt_template_llm = "\n\n [INST] <<SYS>>" + system_prompt + "<</SYS>>\n\n Question: {question} \n\n[/INST]".strip()


    PROMPT_llm = PromptTemplate(template=prompt_template_llm, input_variables=["question"])




    ####### STREAMER FOR TEXT OUTPUT ############
#    model = GlobalVars.get("model")
#    tokenizer = GlobalVars.get("tokenizer")
    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )

    ####### PIPELINE ARGUMENTS FOR THE LLM ############
    ### more info at https://towardsdatascience.com/decoding-strategies-in-large-language-models-9733a8f70539

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )



    ####### ATTACH PIPELINE TO LLM ############

    llm = HuggingFacePipeline(pipeline=text_pipeline)
        

    llmchain = LLMChain(llm=llm, prompt=PROMPT_llm)


    ###### RETRIEVAL QA FROM CHAIN TYPE PARAMS ###########
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT_rag},
#        retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": source_docs_qty}),
        memory=memory,
        verbose=True,
        return_source_documents = True,
        )


    ### Gradio will respond with only 2 arguments from chatbot.interface, first will always be the question, second will be history
    
    ### RAG TOGGLE CHECKBOX 

    if rag_toggle:
    
        response = qa_chain({"query": question})

    else:
    
        response = llmchain({"question": question})
        

    ### 

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)

    return response

#################



chat_interface = gr.ChatInterface(
    
    ### call the main process function above
    
    fn=process_input, 

    ### format the dialogue box, add company avatar image
    
    chatbot = gr.Chatbot(
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname("__file__"), "images/dell-logo-sm.jpg"))),
    ),

    
    additional_inputs=[

                
        gr.Checkbox(label="Use RAG", 
                    value=True, 
                    info="Query LLM directly or query the RAG chain"
                   ),
        
        
        gr.Textbox(label="Persona and role for system prompt:", 
                   lines=3, 
                   value="""Your name is Andie, a helpful concierge at the Dell Tech World conference held in Las Vegas.\
                   Please respond as if you were talking to someone using spoken English language.\
                   The first word of your response should never be Answer:.\
                   You are given a list of helpful information about the conference.\
                   Your goal is to use the given information to answer attendee questions.\
                   Please do not provide any additional information other than what is needed to directly answer the question.\
                   You do not need to show or refer to your sources in your responses.\
                   Please do not make up information that is not available from the given data.\
                   If you can't find the specific information from the given context, please say that you don't know.\
                   Please respond in a helpful, concise manner.\
                   """

                  ),

        gr.Slider(
            label="Number of source docs",
            minimum=1,
            maximum=10,
            step=1,
            value=3,
        ),
        
        gr.Slider(
            label="Max new words (tokens)",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Creativity (Temperature), higher is more creative, lower is less creative:",
            minimum=0.1,
            maximum=1.99,
            step=0.1,
            value=0.6,
        ),
        gr.Slider(
            label="Top probable tokens (Nucleus sampling top-p), affects creativity:",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Number of top tokens to choose from (Top-k):",
            minimum=1,
            maximum=100,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="Repetition penalty:",
            minimum=1.0,
            maximum=1.99,
            step=0.05,
            value=1.2,
        ),
    ],

    
    stop_btn=None,

    examples=[


        ## events csv content
        ["Which booths are found in the showcase floor at Dell Technologies World 2024?"],
        ["What are some common use cases for GenAI?"],
        ["Where is the Charting the Generative AI landscape in healthcare session going to be held?"],
        ["Who is hosting the Understanding GenAI as a workload in a multicloud world session?"],
        ["What enterprise Retrieval Augmented Generation solutions does Dell offer?"],

        ## Powerpoint content
        ["What are some of the results of the Dell Generative AI Pulse Survey?"],
        

        ## pdf content, content creation, workplace productivity
        ["What is Dell's ESG policy in one sentence?"],
        ["Would you please write a professional email response to John explaining the benefits of Dell Powerflex."],
        ["Create a new advertisement for Dell Technologies PowerEdge servers. Please include an interesting headline and product description."],
        ["Create 3 engaging tweets highlighting the key advantages of using Dell Technologies solutions for Generative AI."],
        ["What are the key steps in designing a secure and scalable on-premises solution for GenAI workloads with Dell?"],
        ["Summarize the significant developments from Dell's latest SEC filings."],



    ],

)

#################


theme = gr.themes.Default()


### set width and margins in local css file
### set Title in a markdown object at the top, then render the chat interface

with gr.Blocks(theme=theme, css="style.css", title="Docker Concierge Chat") as demo:
    gr.Markdown(
    """
    # Retrieval Assistant
    """)

    with gr.Tab("Chat Session"):

        chat_interface.render()


    with gr.Tab("Source Citations"):
            
        source_text_box = gr.Textbox(label="Reference Sources")
        get_source_button = gr.Button("Get Source Content")
        get_source_button.click(fn=get_sources, inputs=None, outputs=source_text_box)


    with gr.Tab("Database Files"):


        files_text_box = gr.Textbox(label="Uploaded Files")
        get_files_button = gr.Button("List Uploaded Files")
        get_files_button.click(fn=get_unique_files, inputs=None, outputs=files_text_box)


    with gr.Tab("Model Info"):


        model_info_text_box = gr.Textbox(label="Model Info")
        model_info_button = gr.Button("Get Model Info")
        model_info_button.click(fn=get_model_info, inputs=None, outputs=model_info_text_box)                 


#################

def main():
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    ### call functions
    
    info()
    load_docs()
    prepare_model()

    demo.queue(max_size=5)

    demo.launch(
        share=False,
        debug=True,
        server_name="0.0.0.0",
        allowed_paths=["images/dell-logo-sm.jpg"],
    )


if __name__ == "__main__":
    main()
