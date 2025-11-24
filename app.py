import streamlit as st
from streamlit_mic_recorder import mic_recorder
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Fix for Chinese characters in Matplotlib on Windows
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
import google.generativeai as genai
import numpy as np
import tempfile
import os

# --- Page Config ---
st.set_page_config(
    page_title="嬰兒聲音意圖分析工具",
    page_icon="👶",
    layout="wide"
)

# --- Custom CSS for Warm & Cute UI ---
st.markdown("""
<style>
    .stApp {
        background-color: #FFF5F7; /* Light pink background */
    }
    .main-header {
        font-family: 'Comic Sans MS', 'Chalkboard SE', sans-serif;
        color: #FF8BA7;
        text-align: center;
        font-size: 3em;
        margin-bottom: 20px;
    }
    .sub-header {
        font-family: 'Comic Sans MS', 'Chalkboard SE', sans-serif;
        color: #FFC6C7;
        font-size: 1.5em;
    }
    .stButton>button {
        background-color: #FF8BA7;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #FF6B8B;
    }
    .report-box {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 2px solid #FFC6C7;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<h1 class="main-header">👶 嬰兒聲音意圖分析工具 🍼</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #888;">聽懂寶寶的心聲，給予最溫暖的回應</p>', unsafe_allow_html=True)

# --- Sidebar: API Key ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4529/4529984.png", width=100) # Placeholder cute icon
    st.header("設定")
    
    # Try to get API key from secrets
    try:
        default_api_key = st.secrets.get("GOOGLE_API_KEY", "")
    except FileNotFoundError:
        default_api_key = ""
    except Exception:
        # Handle other potential errors with secrets
        default_api_key = ""
    
    api_key = st.text_input("請輸入 Google API Key", value=default_api_key, type="password", help="我們不會儲存您的 Key，僅用於本次分析。")
    st.info("💡 提示：此工具使用 Gemini AI 來分析寶寶的聲音。")

# --- Helper Functions ---
def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#FF8BA7')
    ax.set_title('聲音波形 (Waveform)', fontsize=12, color='#555')
    ax.set_xlabel('時間 (秒)')
    ax.set_ylabel('振幅')
    plt.tight_layout()
    return fig

def plot_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('聲音頻譜 (Spectrogram)', fontsize=12, color='#555')
    plt.tight_layout()
    return fig

def analyze_audio_with_gemini(audio_file_path, api_key):
    if not api_key:
        return "⚠️ 請先在左側輸入 Google API Key 喔！"
    
    try:
        genai.configure(api_key=api_key)
        
        # Upload the file to Gemini
        myfile = genai.upload_file(audio_file_path)
        
        # Use 'gemini-flash-latest' as it is explicitly listed in the available models.
        model = genai.GenerativeModel("gemini-flash-latest")
        
        prompt = """
        你是一位資深的幼兒教育專家與語言治療師。使用者會上傳一段嬰兒的聲音。
        請分析聲音的音調 (Pitch)、節奏 (Rhythm) 與強度，並判斷嬰兒的潛在意圖（例如：尋求關注、生理需求、社交互動、或是牙牙學語的練習）。
        請用溫暖專業的口吻，條列出分析結果與父母回應建議。
        
        輸出格式建議：
        ### 🔍 聲音分析
        - **音調**: ...
        - **節奏**: ...
        - **強度**: ...
        
        ### 💡 寶寶想說什麼？
        (在此推測寶寶的意圖)
        
        ### ❤️ 建議回應
        (給父母的具體建議)
        """
        
        result = model.generate_content([myfile, prompt])
        return result.text
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        try:
            print("📋 Attempting to list available models for this API Key:")
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(f"   - {m.name}")
        except Exception as list_err:
            print(f"⚠️ Could not list models: {list_err}")
            
        return f"❌ 分析發生錯誤：{str(e)} \n\n (已在終端機列出可用模型，請檢查)"

# --- Main Content ---
col1, col2 = st.columns([1, 1])

audio_source = None
audio_bytes = None
sample_rate = None

with col1:
    st.markdown('<h3 class="sub-header">🎙️ 錄製聲音</h3>', unsafe_allow_html=True)
    # mic_recorder returns a dictionary with 'bytes' if successful
    recorded_audio = mic_recorder(
        start_prompt="開始錄音",
        stop_prompt="停止錄音",
        key='recorder'
    )
    if recorded_audio:
        audio_bytes = recorded_audio['bytes']
        st.audio(audio_bytes, format='audio/wav')
        st.success("錄音完成！")

with col2:
    st.markdown('<h3 class="sub-header">📂 上傳檔案</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("選擇 .wav 或 .mp3 檔案", type=['wav', 'mp3'])
    if uploaded_file:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav') # Streamlit handles mp3 playback with audio/wav hint usually fine, or auto
        st.success("檔案上傳成功！")

# --- Processing & Analysis ---
if audio_bytes:
    st.divider()
    st.markdown('<h3 class="sub-header">📊 聲音視覺化</h3>', unsafe_allow_html=True)
    
    # Save to temp file for librosa and Gemini
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name

    try:
        # Load with Librosa
        y, sr = librosa.load(tmp_file_path)
        
        # Display Plots
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.pyplot(plot_waveform(y, sr))
        with p_col2:
            st.pyplot(plot_spectrogram(y, sr))
            
        st.divider()
        st.markdown('<h3 class="sub-header">🤖 AI 語意分析</h3>', unsafe_allow_html=True)
        
        if st.button("開始分析寶寶的聲音 ✨"):
            with st.spinner("正在聆聽並分析寶寶的聲音...請稍候 🎧"):
                analysis_result = analyze_audio_with_gemini(tmp_file_path, api_key)
                
            st.markdown(f"""
            <div class="report-box">
                {analysis_result}
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"處理音訊時發生錯誤: {e}")
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_file_path):
            # We might want to keep it for a bit if Gemini needs it, but upload_file usually handles it. 
            # However, Gemini file API might need it to persist until inference is done. 
            # Since we wait for response, we can delete now? 
            # Actually, standard practice is to delete after use.
            # But 'upload_file' uploads it to cloud. Local file can be deleted.
            os.unlink(tmp_file_path)

else:
    st.info("👆 請先錄音或上傳檔案，讓我們開始吧！")
