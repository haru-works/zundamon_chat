# 🫛ずんだもん ちゃっと with Zundamon Speech

このリポジトリは、[Zundamon Speech WebUI](https://github.com/zunzun999/zundamon-speech-webui)のWebUIを改造してAIずんだもんと音声チャットを試せるプログラムです。

- 東北ずん子、ずんだもんの公式サイト

公式サイト:https://zunko.jp/

## 下記の技術を使っています。

- [Zundamon Speech WebUI](https://github.com/zunzun999/zundamon-speech-webui)
- LLM：[Google Gemini](https://ai.google.dev/gemini-api/docs?hl=ja)
- LLMフレームワーク：[langchain](https://www.langchain.com/)
- UIフレームワーク：[streamlit](https://streamlit.io/)
- streamlitで音声読み上げする技術の参考資料：[【開発】StreamlitでGeminiを使用したアバター音声対話＆VQAアプリ作ってみた](https://qiita.com/Yuhei0531/items/db894a8fba9c671eb7b0)
- 文章を文節にする技術1：[ja-sentence-segmenter](https://github.com/wwwcojp/ja_sentence_segmenter)
- 文章を文節にする技術2：[sentence-segmenter](https://github.com/mediacloud/sentence-splitter)

## 環境構築
- 1.まずオリジナルのZundamon Speech WebUIを動くようにします
  
　Zundamon Speech WebUIのGithubページを参考にして構築してね
 
  [Zundamon Speech WebUI](https://github.com/zunzun999/zundamon-speech-webui)

- 2.このページからzundamon_webui.pyをダウンロードして、GPT-SoVITSフォルダ配下のzundamon_webui.pyと差し替えます。
  
  上書きすると戻せないから、前のzundamon_webui.pyは名前変えたりしてね。
  
  ![image](https://github.com/user-attachments/assets/e188dbc4-e2bb-45ff-bd32-6f085ba41309)

- 3.Google GeminiのAPIキーを取得します
  
  ※GeminiのAPIキー取得方法は、[Google Gemini](https://ai.google.dev/gemini-api/docs?hl=ja)からやってね

- 4.カレントディレクトリに.envを作って、下の環境変数を追加します。
  ```bash
  GOOGLE_API_KEY=ここにGeminiのAPIキーセット
  ```
  ![image](https://github.com/user-attachments/assets/7c3e01a0-6b3c-4d87-9c38-12c425c479d2)

  ![image](https://github.com/user-attachments/assets/87fefa6f-fa3b-490a-baef-3e77f272a9df)

-5. pythonの追加ライブラリをインストールします。
  ```bash
  pip install pydantic
  pip install langchain_google_genai
  pip install ja_sentence_segmenter
  pip install sentence_splitter
  ```

-6. カレントディレクトリに移動して実行する
  ```bash
  python zundamon_speech_run.py
  ```

-7. 成功したらこんな感じでうごきます

  https://github.com/user-attachments/assets/e4cbb815-fa4a-43c1-8c38-fa3891da5337


# ライセンス情報
このソフトウェアには、次のオープンソースソフトウェアが含まれています。

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (MIT License)
- [GPT-SoVITS Pretrained Models](https://huggingface.co/lj1995/GPT-SoVITS) (MIT License)
- [G2PW Model](https://github.com/GitYCC/g2pW) (Apache 2.0 License)
- [UVR5 (Voice Cleaning)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) (MIT License)
- [Faster Whisper Large V3](https://huggingface.co/Systran/faster-whisper-large-v3) (MIT License)
  
これらは、それぞれのライセンス条項に基づいて提供されます。

Zundamon Voiceモデルのライセンスは次のとおりです 

https://zunko.jp/con_ongen_kiyaku.html
