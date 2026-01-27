import streamlit as st
import whisper
from moviepy import VideoFileClip
import os
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv
import json

# .envファイルから環境変数を読み込む（最新の指定方法を適用）
load_dotenv(".env", override=True)
load_dotenv(".env.local", override=True)

# --- 定数定義 (19個のカテゴリに更新) ---
CATEGORIES = [
    "教育", "経済・金融", "宇宙・物理", "AI・情報", "環境",
    "食料・農業・水産業", "ロボット・技術", "生命・医療",
    "文化・芸術・文芸", "歴史", "国際関係", "都市・建築",
    "人口・社会問題", "数理", "エネルギー・資源",
    "災害・防災", "心理・認知", "政治・法律", "言語・思想"
]

@st.cache_data
def transcribe_video(video_path):
    """動画ファイルから文字起こしを行う"""
    model = whisper.load_model("base") 
    result = model.transcribe(video_path, fp16=False)
    return result['text']

def generate_analysis(text, api_key, num_keywords=10):
    """テキストから要約、キーワード、関連カテゴリ（最大2つ）を抽出する"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-flash-latest')
        
        category_list_str = ", ".join(CATEGORIES)
        
        prompt = f'''
        以下の文章は、ある講義を文字起こししたものです。

        ### 指示 ###
        1. この講義内容を200文字程度で要約してください。
        2. 要約内容から、最も関連性の高いキーワードを{num_keywords}個抽出してください。
        3. 以下のカテゴリリストから、この講義内容に最も関連するものを**最大2つまで**選んでください。
        4. 結果を必ず以下のJSON形式で出力してください。
        {{
          "summary": "（ここに要約文）",
          "keywords": ["キーワード1", "キーワード2", ...],
          "categories": ["カテゴリA", "カテゴリB", ...]
        }}

        ### カテゴリリスト ###
        {category_list_str}

        ### 文章 ###
        ---
        {text}
        ---
        '''
        
        response = model.generate_content(prompt)
        
        # モデルからの応答テキストを抽出し、JSONとしてパースする
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        analysis_result = json.loads(response_text)

        # プログラム側でも最大2つであることを保証する
        if "categories" in analysis_result:
            analysis_result["categories"] = analysis_result["categories"][:2]
        
        return analysis_result

    except Exception as e:
        st.error(f"Gemini APIとの連携中または結果の解析中にエラーが発生しました: {e}")
        return None


# --- Streamlit UI ---

st.set_page_config(page_title="動画からキーワード抽出", layout="wide")
st.title("講義動画の文字起こし＆キーワード抽出")
st.info("講義の動画ファイルをアップロードすると、AIが自動で文字起こしを行い、内容を要約して関連キーワードを抽出します。")

# APIキーを環境変数から取得
api_key = os.getenv("API_KEY")

if not api_key:
    st.error("エラー: Gemini APIキーが設定されていません。.envファイルを確認してください。")
else:
    # ffmpegのチェック（Windows環境用）
    if not os.path.exists("ffmpeg.exe") and os.name == 'nt':
        st.warning("注: `ffmpeg.exe` が見つかりません。パスが通っていない場合は動画処理に失敗する可能性があります。")

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
            
            status_text.text("Step 2/3: 音声を文字起こししています...")
            transcribed_text = transcribe_video(audio_path)
            progress_bar.progress(66)

            status_text.text("Step 3/3: AIによる分析（要約・キーワード・カテゴリ最大2つ）を実行中...")
            analysis_result = generate_analysis(transcribed_text, api_key, num_keywords=10)
            progress_bar.progress(100)

            if analysis_result:
                status_text.success("処理が完了しました！")
                
                st.subheader("関連カテゴリ (最大2つ)")
                st.markdown(f"#### **{ '、'.join(analysis_result['categories']) }**")

                st.subheader("AIによる要約")
                st.info(analysis_result['summary'])

                st.subheader("抽出されたキーワード")
                st.markdown(f"##### { '、'.join(analysis_result['keywords']) }")
                
                with st.expander("全文文字起こし結果を見る"):
                    st.text_area("", transcribed_text, height=300)
            else:
                status_text.error("分析結果の取得に失敗しました。")

        except Exception as e:
            st.error(f"処理中にエラーが発生しました: {e}")
        
        finally:
            # 後処理: メモリ解放と一時ファイルの削除
            if 'video_clip' in locals():
                video_clip.close()
            if 'video_path' in locals() and os.path.exists(video_path):
                os.remove(video_path)
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)