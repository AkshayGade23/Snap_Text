import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from huggingsound import SpeechRecognitionModel
import speech_recognition as sr
from htmlTemplates import css, bot_template, user_template,sidebar_custom_css
from pytube import YouTube
import librosa
import soundfile as sf
import torch
from transformers import pipeline
import shutil
from streamlit_chat import message
from gtts import gTTS
from io import BytesIO
import pyttsx3 
from googletrans import Translator


def get_pdf_text(pdf_docs):
    text = ""
    for file in pdf_docs:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file.type.startswith("image/"):
            img = Image.open(file)
            pytesseract.pytesseract.tesseract_cmd = f"{os.environ['TESSERACT_CMD']}"
            text += pytesseract.image_to_string(img)
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

def translate_text(text, response_language):
    translator =Translator()
    translated_text = translator.translate(text, dest=response_language)
    return translated_text.text

def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vetorestore):
    # llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", model_kwargs={"temperature":5,
                                                    #    "max_length":64})
    # llm=pipeline('text-generation', model='gpt2-medium')
    
    llm_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceEndpoint(
            repo_id=llm_model, 
            # model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k, "load_in_8bit": True}
            temperature = 0.7,
            max_new_tokens =1024,
            top_k = 3,
            load_in_8bit = True,
        )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question, target_language,response_language):
    # Translate user's question to English (or desired target language)
    translated_question = translate_text(user_question, target_language)

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))

    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
            
    #     else:
    #         # Translate bot's response back to the user's language
    #         translated_response = translate_text(message.content, response_language)
    #         bot_response = bot_template.replace("{{MSG}}", translated_response)
    #         st.write(bot_response, unsafe_allow_html=True)
    

            # Add a "Talk Back" button for each bot response
            talk_back_button_key = f"talk_back_button_{i}"
            talk_back_button = st.button(f"Talk Back", key=talk_back_button_key)

            # If the "Talk Back" button is clicked, read out the bot response
            if talk_back_button:
                engine = pyttsx3.init()
                engine.say(messages.content) 
                engine.runAndWait()


def handle_microphone_input():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Say something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        st.session_state.microphone_text = recognizer.recognize_google(audio)
        st.session_state.user_question = st.session_state.microphone_text
    except sr.UnknownValueError:
        st.session_state.user_question = "Could not understand audio"
    except sr.RequestError as e:
        st.session_state.user_question = f"Error with the speech recognition service; {e}"



def download_video(videoURL):
    
    # deleting exiting directories \audio and \audio_chunks
    if os.path.exists("audio"):
        shutil.rmtree("audio")

    if os.path.exists("audio_chunks"):
        shutil.rmtree("audio_chunks")

    # making directories \audio and \audio_chunks
    os.mkdir("./audio") # .mp3 and .wav file of whole video
    os.mkdir("./audio_chunks") # original files chunks
    print("dfsdfsdafsdfadsf")
    print(videoURL)
    yt = YouTube(videoURL)

    yt.streams.filter(only_audio = True, file_extension = 'mp4').first().download(filename = 'audio\ytAudio.mp4')
    
    os.system("ffmpeg -i audio\ytAudio.mp4 -acodec pcm_s16le -ar 16000 audio\ytAudio.wav")
   


def audio_chunking(inputFile):
    librosa.get_samplerate(inputFile)
    # Stream over 30 seconds chunks rather than load the whole file
    stream = librosa.stream(
        inputFile,
        block_length=30,
        frame_length=16000,
        hop_length=16000
    )

    number_of_files = 0
    for i,speech in enumerate(stream):
        number_of_files +=1
        sf.write(f'audio_chunks\{i}.wav', speech, 16000)
    
    return number_of_files


def audio_transcribe(number_of_files):
    audioPath = []
    
    for a in range(number_of_files):
        audioPath.append(f'audio_chunks\{a}.wav')

    device = "cuda" if torch.cuda.is_available() else "cpu"
   

    #  issue with "from huggingsound import SpeechRecognitionModel"
    model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english", device = device)

    transcriptions = model.transcribe(audioPath)

    fullTranscript = ' '
    for item in transcriptions:
        fullTranscript += ''.join(item['transcription'])


    return fullTranscript


def youtube_process(videoURL):
    # downloading video and stored in /cotent/s
    download_video(videoURL)
    # audio chunking 
    inputFile = "audio\ytAudio.wav"
    number_of_files= audio_chunking(inputFile)
    # # audio transcribe and summarization
    summarizedText = audio_transcribe(number_of_files)
    print(summarizedText)
    return summarizedText   



def main():
    load_dotenv()
    st.set_page_config(page_title="Snap Text",
                       page_icon=":bot:")
    # st.write(css, unsafe_allow_html=True)
    # st.markdown(sidebar_custom_css, unsafe_allow_html=True)
    
    target_language = st.selectbox("Select Document language (Target)", ["en", "es", "fr", "de"], key="target_language")
    response_language = st.selectbox("Select Document language (Response)", ["en", "es", "fr", "de"], key="response_language")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
    if "youtube_link" not in st.session_state:
        st.session_state.youtube_link = ""
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    st.header("Snap Text :books:")

    # Add a microphone button
    # if st.button("🎤 Microphone"):
    #     handle_microphone_input()

    # user_question = st.text_input("Ask a question about your documents:", value=st.session_state.user_question)

    # if  st.session_state.processComplete == True:
    #     user_question = st.chat_input("Ask Question about your files.")
        
    #     if user_question:
    #         handle_userinput(user_question,target_language,response_language)

    with st.sidebar:

        # Pdf and documents
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process Pdf'", accept_multiple_files=True)
        

        if st.button("Process Pdf"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.processComplete = True
       
        st.header("OR")
        
        # Youtube link
        st.subheader("Youtube Link")
        youtube_link = st.text_input("Insert video link and click on 'Process Link'")
        if st.button("Process Link"):
            with st.spinner("Processing"):
                # get transcription of youtube video
                raw_text = youtube_process(youtube_link)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                st.session_state.processComplete = True
    
       

                

        
    if  st.session_state.processComplete == True:
        user_question = st.chat_input("Ask Question about your files.")
    
        if user_question:
            handle_userinput(user_question,target_language,response_language)

        

if __name__ == '__main__':
    main()
