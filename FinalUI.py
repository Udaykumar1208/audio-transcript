import os
import whisper
import streamlit  as st
from pydub import AudioSegment
import openai
from dotenv import load_dotenv
from googletrans import Translator
from keybert import KeyBERT
from transformers import pipeline


from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

load_dotenv()

import googletrans

translator = googletrans.Translator()

st.set_page_config(
    page_title="Elite Notes",
    page_icon=":note:",
    layout="wide",
    initial_sidebar_state="auto",
)

upload_path = "uploads/"
download_path = "new/"
transcript_path = "transcripts/"

@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def to_mp3(audio_file, output_audio_file, upload_path, download_path):
    ## Converting Different Audio Formats To MP3 ##
    if audio_file.name.split('.')[-1].lower()=="wav":
        audio_data = AudioSegment.from_wav(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3")

    elif audio_file.name.split('.')[-1].lower()=="mp3":
        audio_data = AudioSegment.from_mp3(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3")

    elif audio_file.name.split('.')[-1].lower()=="ogg":
        audio_data = AudioSegment.from_ogg(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3")

    elif audio_file.name.split('.')[-1].lower()=="wma":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"wma")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3")

    elif audio_file.name.split('.')[-1].lower()=="aac":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"aac")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3")

    elif audio_file.name.split('.')[-1].lower()=="m4a":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"m4a")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3")

    elif audio_file.name.split('.')[-1].lower()=="mp4":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"mp4")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3")
    return output_audio_file

@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def process_audio(filename, model_type):
    model = whisper.load_model(model_type)
    result = model.transcribe(filename)
    return result["text"]

@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def save_transcript(transcript_data, txt_file):
    with open(os.path.join(transcript_path, txt_file),"w") as f:
        f.write(transcript_data)

st.markdown("<h1 style='text-align: center; color: red; font-size:55px; '>Elite Notes</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='padding: 10px; text-align: center; color: lightblack; font-size: 15px; margin : 15px auto;'>Upload Your Audio Files, Get The Transcription And Save Your Time</h1>", unsafe_allow_html=True)


st.markdown("---")
st.markdown("<h3 style= 'color: red;'>Audio Transcribe</h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload audio file", type=["wav","mp3","ogg","wma","aac","m4a","mp4","flv"])

audio_file = None

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
        f.write((uploaded_file).getbuffer())
    with st.spinner(f"Processing Audio ... 💫"):
        output_audio_file = uploaded_file.name.split('.')[0] + '.mp3'
        output_audio_file = to_mp3(uploaded_file, output_audio_file, upload_path, download_path)
        audio_file = open(os.path.join(download_path,output_audio_file), 'rb')
        audio_bytes = audio_file.read()
    print("Opening ",audio_file)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.audio(audio_bytes)
    with col2:
        whisper_model_type = st.radio("Please choose your model type", ('Tiny', 'Base', 'Small', 'Medium', 'Large'))

    st.markdown("<h1 style='text-align: center; color: red; '>Features</h1>", unsafe_allow_html=True)
    
    if st.checkbox("Generate Transcript "):
        st.markdown("<h3 style= 'color: red;'>Transcript:</h3>", unsafe_allow_html=True)
        with st.spinner(f"Generating Transcript... 💫"):
            transcript = process_audio(str(os.path.abspath(os.path.join(download_path,output_audio_file))), whisper_model_type.lower())
            
            if transcript is not None:
                
                st.markdown(f'<p style="padding: 10px; text-align: left; color:lightblack; background:#FFFACD ; font-size:15px; margin : 15px auto;">{transcript}</p>', unsafe_allow_html=True)
                st.download_button('Download Transcript', transcript, 'Transcript.txt')
#---------------------------------------------------------------------------------------------




@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def process_audio(filename, model_type):
    model = whisper.load_model(model_type)
    result = model.transcribe(filename)
    return result["text"]

@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def save_transcript(transcript_data, txt_file):
    with open(os.path.join(transcript_path, txt_file),"w") as f:
        f.write(transcript_data)

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
        f.write((uploaded_file).getbuffer())
    with st.spinner(f"Processing Audio ... 💫"):
        output_audio_file = uploaded_file.name.split('.')[0] + '.mp3'
        output_audio_file = to_mp3(uploaded_file, output_audio_file, upload_path, download_path)
        audio_file = open(os.path.join(download_path,output_audio_file), 'rb')
        audio_bytes = audio_file.read()
    print("Opening ",audio_file)
   

    

    
    if st.checkbox("Generate Translate "):
        st.markdown("<h3 style= 'color: red;'>Translation:</h3>", unsafe_allow_html=True)
        with st.spinner(f"Generating Translate... 💫"):
            
            transcript = process_audio(str(os.path.abspath(os.path.join(download_path,output_audio_file))), whisper_model_type.lower())
           
            translated_text = translator.translate(transcript, dest='hi').text
            
            st.markdown(f'<p style="padding: 10px; text-align: left; color:black; background:#FFA500 ; font-size:15px; margin : 15px auto;">{translated_text}</p>', unsafe_allow_html=True)
           
#----------------------------------------------------------------------------------------------------------------------------------------------


@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def process_audio(filename, model_type):
    model = whisper.load_model(model_type)
    result = model.transcribe(filename)
    return result["text"]

@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def save_transcript(transcript_data, txt_file):
    with open(os.path.join(transcript_path, txt_file),"w") as f:
        f.write(transcript_data)
        
        
def point_wise_summary(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=1050, min_length=30)
    summary_text = summary[0]["summary_text"]
    points = summary_text.split(".")
    return [point   for point in points]


if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
        f.write((uploaded_file).getbuffer())
    with st.spinner(f"Processing Audio ... 💫"):
        output_audio_file = uploaded_file.name.split('.')[0] + '.mp3'
        output_audio_file = to_mp3(uploaded_file, output_audio_file, upload_path, download_path)
        audio_file = open(os.path.join(download_path,output_audio_file), 'rb')
        audio_bytes = audio_file.read()
    print("Opening ",audio_file)
   

    

    
    if st.checkbox("summarization"): 
        st.markdown("<h3 style= 'color: red;'>Summarization</h3>", unsafe_allow_html=True)
        with st.spinner(f"Generating Summary... 💫"):
            
            text = process_audio(str(os.path.abspath(os.path.join(download_path,output_audio_file))), whisper_model_type.lower())
            
                    
            # Show summary
            if text:
                summary_points = point_wise_summary(text)
                
                for point in summary_points:
                    st.markdown(f'<p style="padding: 10px; text-align: left; color:lightblack; background:#FFFACD ; font-size:15px; margin : 15px auto;">{"❖" + point}</p>', unsafe_allow_html=True)
                    
        #-----------------------------------------------------------------------------------------------------            
           



@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def process_audio(filename, model_type):
    model = whisper.load_model(model_type)
    result = model.transcribe(filename)
    return result["text"]

@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def save_transcript(transcript_data, txt_file):
    with open(os.path.join(transcript_path, txt_file),"w") as f:
        f.write(transcript_data)

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
        f.write((uploaded_file).getbuffer())
    with st.spinner(f"Processing Audio ... 💫"):
        output_audio_file = uploaded_file.name.split('.')[0] + '.mp3'
        output_audio_file = to_mp3(uploaded_file, output_audio_file, upload_path, download_path)
        audio_file = open(os.path.join(download_path,output_audio_file), 'rb')
        audio_bytes = audio_file.read()
    print("Opening ",audio_file)
   
    
    if st.checkbox("Generate Keywords "):
        st.markdown("<h3 style= 'color: red;'>Keyword Extraction</h3>", unsafe_allow_html=True)
        with st.spinner(f"Generating Keywords... 💫"):
            
            
            transcript = process_audio(str(os.path.abspath(os.path.join(download_path,output_audio_file))), whisper_model_type.lower())
            
             

            kw_model = KeyBERT()
            keywords = kw_model.extract_keywords(transcript)
            for items in keywords:
  
                 keys = st.write("▣ ",items[0])
                
                


    
    

    

            