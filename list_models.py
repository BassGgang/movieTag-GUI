
import google.generativeai as genai
import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# APIキーを環境変数から取得
api_key = os.getenv("API_KEY")

if not api_key:
    print("エラー: APIキーが .env ファイルに設定されていません。")
else:
    try:
        genai.configure(api_key=api_key)
        
        print("現在利用可能で、'generateContent'をサポートしているモデルは以下の通りです。")
        print("---")
        
        model_found = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"・ {m.name}")
                model_found = True
        
        if not model_found:
            print("利用可能なモデルが見つかりませんでした。")
            
    except Exception as e:
        print(f"APIへの接続中にエラーが発生しました: {e}")

print("---")
