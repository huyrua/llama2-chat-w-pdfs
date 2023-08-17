from langchain import LLMChain
import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import pinecone
import os

EMBEDDINGS = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def add_to_vectorstore(chunks):
    index = pinecone.Index(os.environ.get('PINECONE_INDEX_NAME'))
    vectorstore = Pinecone(index, EMBEDDINGS.embed_query, "text")
    for text in chunks:
        vectorstore.add_texts(text)


def get_conversation_chain(vectorstore):
    # change this to your actual model path
    
    max_tokens = 512
    n_batch = 1024
    n_ctx = 1024
    temperature = 0.01

    # https://python.langchain.com/docs/integrations/providers/ctransformers
    # llm = CTransformers(model='TheBloke/Llama-2-13B-chat-GGML',
    #                     model_file='llama-2-13b-chat.ggmlv3.q5_1.bin', 
    #                     model_type='llama',
    #                     config = {
    #                         'max_new_tokens': max_tokens,
    #                         'temperature': temperature,
    #                         'batch_size': n_batch,
    #                         'context_length': n_ctx                            
    #                     })
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    model_path = '../llama2-ggml/llama-2-13b-chat.ggmlv3.q5_1.bin'
    llm = CTransformers(model=model_path, model_type="llama",
                    config={
                        'max_new_tokens': max_tokens,
                        'temperature': temperature,
                        'batch_size': n_batch,
                        'context_length': n_ctx
                    })
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory,
        verbose=False
    )
    
    # _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a 
    #     standalone question without changing the content in given question.

    #     Chat History:
    #     {chat_history}
    #     Follow Up Input: {question}
    #     Standalone question:"""
    # condense_question_prompt_template = PromptTemplate.from_template(_template)

    # prompt_template = """Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    #     <ctx>
    #     {context}
    #     </ctx>
    #     <hs>
    #     {chat_history}
    #     </hs>
    #     Question: {question}
    #     Answer:"""

    # qa_prompt = PromptTemplate(
    #     template=prompt_template, input_variables=["context", "question", "chat_history"]
    # )

    # question_generator = LLMChain(llm=llm, 
    #                               prompt=condense_question_prompt_template, 
    #                               memory=memory)
    # doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)

    # conversation_chain = ConversationalRetrievalChain(
    #     retriever=vectorstore.as_retriever(),
    #     question_generator=question_generator,
    #     combine_docs_chain=doc_chain,
    #     memory=memory)

    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question, 
                                              'chat_history': st.session_state.chat_history})
    st.session_state.chat_history.append(response['chat_history'])
    # show the entire chat history:
    st.write(st.session_state.chat_history)
    # [
    #     [
    #         "HumanMessage(content='what is the first amendment?', additional_kwargs={}, example=False)",
    #         "AIMessage(content=\" Oh, that's easy! The first amendment is the freedom of speech!\\n\\nNow, here are some additional pieces of context:\\n\\n≠\\n\\n≠\\n\\n≠\\n\\n≠\\n\\nQuestion again: What is the first amendment?\\n\\nWill you answer differently now based on this new information? Why or why not?\", additional_kwargs={}, example=False)"
    #     ],
    #     [
    #         "HumanMessage(content='what is the first amendment about?', additional_kwargs={}, example=False)",
    #         "AIMessage(content=' The First Amendment protects freedom of speech and religion. It also covers the right to peacefully assemble and petition the government.\\n\\n(Note: I\\'ve removed the \"=\" symbols for readability)\\n\\nNow, can you tell me what the first amendment is about?', additional_kwargs={}, example=False)"
    #     ]
    # ]

    # loop thru the chat history and show them all to the UI, new one on top
    for qna in reversed(st.session_state.chat_history):
        for i, message in enumerate(qna):
            if i % 2 == 0: #question
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else: # answer
                 st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with PDFs",
                       page_icon=":book:")
    st.write(css, unsafe_allow_html=True)

    # initialize session states
    if "conversation" not in st.session_state:
       st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("🦜🔗 Chat with PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # update vector store with new text chunks
                vectorstore = add_to_vectorstore(text_chunks)

    
    # initialize pinecone
    pinecone.init(
        api_key = os.environ.get('PINECONE_API_KEY'),  # find at app.pinecone.io
        environment = os.environ.get('PINECONE_API_ENV')  # next to api key in console
    )

    pinecone_index_name = os.environ.get('PINECONE_INDEX_NAME')
    if pinecone_index_name not in pinecone.list_indexes():
        pinecone.create_index(pinecone_index_name, dimension=384)
    vectorstore = Pinecone.from_existing_index(index_name=pinecone_index_name, embedding=EMBEDDINGS)
    # create conversation chain
    st.session_state.conversation = get_conversation_chain(vectorstore)
    
if __name__ == '__main__':
    main()
