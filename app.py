import streamlit as st
import whisper
from moviepy import VideoFileClip, AudioFileClip, AudioArrayClip, vfx, concatenate_videoclips
import os
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# --- 関数定義 ---

@st.cache_data
def transcribe_video(video_path):
    """動画ファイルから文字起こしを行う"""
    model = whisper.load_model("base") 
    result = model.transcribe(video_path, fp16=False)
    return result['text']

def extract_keywords(text, api_key, num_keywords=10):
    """テキストからキーワードを生成する"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-flash-latest')
        
        prompt = f'''
        以下の文章は、ある講義を文字起こししたものです。
        この講義内容を要約し、最も関連性の高いキーワードを{num_keywords}個、カンマ区切りで生成してください。
        キーワードは単語または短いフレーズとし、説明は含めないでください。

        ---
        {text}
        ---
        '''
        
        response = model.generate_content(prompt)
        if response.text:
            keywords = [keyword.strip() for keyword in response.text.split(',')]
            return keywords
        else:
            st.error("キーワードを生成できませんでした。モデルからの応答が空です。")
            return ["キーワードの生成に失敗しました。"]

    except Exception as e:
        st.error(f"Gemini APIとの連携中にエラーが発生しました: {e}")
        return ["キーワードの生成に失敗しました。"]


# --- Streamlit UI ---

st.set_page_config(page_title="動画からキーワード抽出", layout="wide")
st.title("講義動画の文字起こし＆キーワード抽出")
st.info("講義の動画ファイルをアップロードすると、AIが自動で文字起こしを行い、内容を要約して関連キーワードを10個生成します。")

# APIキーを環境変数から取得
api_key = os.getenv("API_KEY")

# APIキーが設定されているか確認
if not api_key:
    st.error("エラー: Gemini APIキーが設定されていません。.envファイルに「API_KEY='ご自身のAPIキー'」を追記してください。")
else:
    # ffmpeg.exeが同じディレクトリにあることを確認
    if not os.path.exists("ffmpeg.exe"):
        st.error("エラー: `ffmpeg.exe` が見つかりません。アプリケーションと同じフォルダに配置してください。")
    else:
        uploaded_file = st.file_uploader(
            "動画ファイルを選択してください (MP4, MOV, AVI, M4Aなど)", 
            type=['mp4', 'mov', 'avi', 'm4a', 'mpeg', 'wav', 'mp3']
        )

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                video_path = tmp_file.name

            try:
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Step 1/3: 動画から音声を抽出しています...")
                video_clip = VideoFileClip(video_path)
                audio_path = video_path + ".wav"
                video_clip.audio.write_audiofile(audio_path)
                progress_bar.progress(33)
                
                status_text.text("Step 2/3: 音声を文字起こししています...（時間がかかる場合があります）")
                transcribed_text = transcribe_video(audio_path)
                progress_bar.progress(66)

                status_text.text("Step 3/3: テキストからキーワードを抽出しています...")
                keywords = extract_keywords(transcribed_text, api_key, num_keywords=10)
                progress_bar.progress(100)

                status_text.success("処理が完了しました！")
                
                st.subheader("抽出されたキーワード")
                st.markdown(f"### **{ '、'.join(keywords) }**")
                
                with st.expander("全文文字起こし結果を見る"):
                    st.text_area("", transcribed_text, height=300)

            except Exception as e:
                st.error(f"処理中にエラーが発生しました: {e}")
                st.error("動画ファイルが破損しているか、対応していない形式の可能性があります。")
            
            finally:
                if 'video_clip' in locals():
                    video_clip.close()
                if 'video_path' in locals() and os.path.exists(video_path):
                    os.remove(video_path)
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.remove(audio_path)
