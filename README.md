# zundamon_chat
ずんだもんちゃっと

## 環境構築
- 1.まずオリジナルのZundamon Speech WebUIを動くようにします
  下記のページを参考に構築してね
  https://github.com/zunzun999/zundamon-speech-webui

- 2. このページからzundamon_webui.pyをダウンロードして、GPT-SoVITSフォルダ配下のzundamon_webui.pyと差し替えます。
     上書きすると戻せないから、前のzundamon_webui.pyは名前変えたりしてね。
![image](https://github.com/user-attachments/assets/e188dbc4-e2bb-45ff-bd32-6f085ba41309)


- 3.Google GeminiのAPIキーを取得します
※GeminiのAPIキー取得方法は、https://ai.google.dev/gemini-api/docs?hl=ja からやってね

- 4.カレントディレクトリに.envを作って、下の環境変数を追加します。
```bash
GOOGLE_API_KEY=ここにGeminiのAPIキーセット
```
![image](https://github.com/user-attachments/assets/7c3e01a0-6b3c-4d87-9c38-12c425c479d2)

こんな感じでGOOGLE_API_KEYを設定してね
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

