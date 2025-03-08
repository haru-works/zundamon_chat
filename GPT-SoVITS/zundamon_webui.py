import streamlit as st
import tempfile
import os
import soundfile as sf
import base64
import time
from pydantic import SecretStr

# langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,SystemMessage

# 文章分割
import functools
from ja_sentence_segmenter.common.pipeline import make_pipeline
from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation
from sentence_splitter import SentenceSplitter

import sys

# 追加
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


# 環境変数の読込
from dotenv import load_dotenv
load_dotenv()


current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.insert(0, repo_root)
sys.path.insert(0, current_dir)
sys.path.append(os.path.join(current_dir, 'GPT_SoVITS'))

# 推論関数と必要なパッケージをインポートします（必要に応じてインポートパスを調整します）
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

# モデルファイルパスを修正しました（必要に応じて変更してください）
GPT_MODEL_PATH =  os.path.join(current_dir, 'GPT_weights_v2', 'zudamon_style_1-e15.ckpt')
SOVITS_MODEL_PATH = os.path.join(current_dir, 'SoVITS_weights_v2', 'zudamon_style_1_e8_s96.pth')

# メッセージ
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# session_stateを使用して生成されたオーディオデータを保存して、再実行のクリアを防ぐ
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = []


# 推論関数を定義
def synthesize(ref_audio_path, ref_text_path, ref_language, target_text_path, target_language, output_path):

    # 参照テキストを読んでください
    with open(ref_text_path, 'r', encoding='utf-8') as file:
        ref_text = file.read()

    # ターゲットテキストを読んでください
    with open(target_text_path, 'r', encoding='utf-8') as file:
        target_text = file.read()

    # モデルの重み変更し
    change_gpt_weights(gpt_path=GPT_MODEL_PATH)
    change_sovits_weights(sovits_path=SOVITS_MODEL_PATH)

    # オーディオを生成します
    synthesis_result = get_tts_wav(ref_wav_path=ref_audio_path,
                                   prompt_text=ref_text,
                                   prompt_language=ref_language,
                                   text=target_text,
                                   text_language=target_language,
                                   top_p=0.6,
                                   temperature=0.6)

    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        output_wav_path = os.path.join(output_path, "output.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"音声ファイル保存: {output_wav_path}")


# llm初期化
def init_llm(llm_model: str):
    if llm_model == 'gemini-1.5-flash' or llm_model == 'gemini-2.0-flash':
        return ChatGoogleGenerativeAI(model=llm_model,
                                      api_key=SecretStr(os.getenv('GOOGLE_API_KEY')),
                                      temperature=1.0,)
    else:
        st.error(f'サポートされてないLLMモデルなのだ: {llm_model}',icon="✖")



# メッセージ
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# session_stateを使用して生成されたオーディオデータを保存して、再実行のクリアを防ぐ
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = []


# 音声再生
def play_audio(output_wav_path,assistant_avatar,sentences):
    with open(output_wav_path, "rb") as fw:
        audio_bytes_data = fw.read()
        fw.close()
        audio_placeholder = st.empty()
        audio_str = "data:audio/ogg;base64,%s"%(base64.b64encode(audio_bytes_data).decode())
        audio_html = f"""
                        <audio autoplay=True>
                        <source src={audio_str} type="audio/ogg" autoplay=True>
                        Your browser does not support the audio element.
                        </audio>
                    """
        audio_placeholder.empty()
        time.sleep(0.5)
        audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
        st.markdown(sentences,unsafe_allow_html=True)
        st.session_state["messages"].append({"role": "assistant", "avatar": assistant_avatar,"content": sentences})
        st.audio(audio_bytes_data, format="audio/wav")


# システムプロンプト生成
def generate_system_prompt(target_language,max_text_count):
    system_prompt=f"""
    あなたはずんだもんです。
    ユーザーの質問に制約条件に従って親切に回答してください。

    # 制約条件
        - 絵文字は使わない事。
        - {max_text_count}文字以内で答える事。
        - 回答言語で必ず回答する事。

    # 回答言語：{target_language}

    """
    # システムプロンプト
    return  SystemMessage(
                        content=[
                            {
                                "type": "text",
                                "text": system_prompt,
                            },
                        ]
    )


# ユーザープロンプト生成
def generate_user_prompt(prompt):
    # ユーザープロンプト
    return HumanMessage(
                    content=[
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ]
                    )



# ページ構成
st.set_page_config(page_title="ずんだもん ちゃっと with Zundamon Speech",page_icon="🫛", layout="wide")

# タイトル
st.markdown("# 🫛ずんだもん ちゃっと with Zundamon Speech", unsafe_allow_html=True)

# チャットのアバター
user_avatar = "😀"
assistant_avatar = "🫛"

# サイドバータイトル
st.sidebar.markdown("# 📝ちゃっと設定")

# LLMモデル選択
st.sidebar.markdown("### 🤖LLMモデル")
llm_model = st.sidebar.radio("LLMモデル選択するのだ",["gemini-1.5-flash","gemini-2.0-flash"], index=0)

# リファレンス音声ファイル選択
st.sidebar.markdown("### 🔊リファレンス音声ファイル")
uploaded_audio = st.sidebar.file_uploader("リファレンス音声ファイルをアップロードするのだ (WAV, FLAC, MP3)", type=["wav", "flac", "mp3"])

# リファレンステキスト入力
st.sidebar.markdown("### 📄リファレンステキスト")
default_target_text = "流し切りが完全に入ればデバフの効果が付与される"
default_ref_text = st.sidebar.text_area("リファレンステキストを入力するのだ", value=default_target_text, height=100)

# アップロードされたリファレンスオーディオを一時ファイルとして保存（元の拡張機能を維持）
if uploaded_audio is not None:
    audio_suffix = os.path.splitext(uploaded_audio.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix) as tmp_audio:
        tmp_audio.write(uploaded_audio.read())
        tmp_audio_path = tmp_audio.name

 # 参照テキストを一時ファイルに書き込み
if default_ref_text != "":
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as tmp_ref_text:
        tmp_ref_text.write(default_ref_text)
        tmp_ref_text_path = tmp_ref_text.name

# チャット表示コンテナ
main_container = st.container(height=500)
# チャット表示
for message in st.session_state["messages"]:
    with main_container.chat_message(name=message["role"],avatar=message["avatar"]):
        # メッセージ表示
        st.markdown(message["content"],unsafe_allow_html=True)

# ボタンコンテナ
buttons_container = st.container()

# チャット入力
prompt = buttons_container.text_area(label="質問を入力するのだ",
                                    placeholder="ずんだもんの事教えて",
                                    height=80)

# カラム生成
lang_selectbox,run_button,blank,new_chat_button = buttons_container.columns([1,1,3,1],vertical_alignment="bottom")


# 言語選択
with lang_selectbox:
    target_language = st.selectbox("言語を選択するのだ", ["Japanese",
                                                        "English",
                                                        "Chinese",
                                                        "Cantonese",
                                                        "Korean",])


# ちゃっと実行ボタン
with run_button:
    if st.button(label="ちゃっと実行",icon="💭",type="primary"):
        if prompt == "":
            main_container.error("プロンプトを入力するのだ",icon="✖")
            st.stop()

        if uploaded_audio is None:
            main_container.error("リファレンス音声をアップロードするのだ",icon="✖")
            st.stop()

        else:
            # ユーザー
            with main_container.chat_message("user", avatar=user_avatar):
                # システムプロンプト
                system_prompt = generate_system_prompt(target_language,30)
                # ユーザープロンプト
                user_prompt = generate_user_prompt(prompt)
                # 表示用
                st.session_state["messages"].append({"role": "user", "avatar": user_avatar,"content":  prompt})
                st.markdown(prompt,unsafe_allow_html=True)
                # AI
                with main_container.chat_message("assistant", avatar=assistant_avatar):
                    try:
                        with st.spinner(text="💭処理中なのだ...",show_time=True):
                            # LLM初期化
                            llm = init_llm(llm_model)
                            # LLM実行
                            response = llm.invoke([system_prompt,user_prompt])
                            # 結果取得
                            result = response.content
                            # 結果の文章を文節ごとに分割
                            split_sentences = []
                            if target_language == "Japanese":
                                split_punc2 = functools.partial(split_punctuation,
                                                                punctuations=r"。!?")
                                concat_tail_te = functools.partial(concatenate_matching,
                                                                   former_matching_rule=r"^(?P<result>.+)(て)$",
                                                                   remove_former_matched=False)
                                segmenter = make_pipeline(normalize,
                                                          split_newline,
                                                          concat_tail_te,
                                                          split_punc2)
                                split_sentences = list(segmenter(result))
                            else:
                                splitter = SentenceSplitter(language='en')
                                split_sentences = splitter.split(text=result)

                            # デバッグ用
                            print("target_language------------------------------")
                            print(target_language)
                            print("split_sentences------------------------------")
                            print(split_sentences)
                            print("---------------------------------------------")

                            for sentences in split_sentences:
                                # ターゲットテキストを一時ファイルに書込
                                with tempfile.NamedTemporaryFile(delete=False,
                                                                 suffix=".txt",
                                                                 mode='w',
                                                                 encoding='utf-8') as tmp_target_text:
                                    tmp_target_text.write(sentences)
                                    tmp_target_text_path = tmp_target_text.name
                                # 一時出力ディレクトリ作成
                                tmp_output_dir = tempfile.mkdtemp()
                                # 推論関数を呼び出し
                                ref_language = "Japanese"
                                synthesize(tmp_audio_path,
                                            tmp_ref_text_path,
                                            ref_language,
                                            tmp_target_text_path,
                                            target_language,
                                            tmp_output_dir)
                                # 推論関数は、output.wavという名前のオーディオファイルを生成
                                output_wav_path = os.path.join(tmp_output_dir, "output.wav")
                                if os.path.exists(output_wav_path):
                                    # 音声再生
                                    play_audio(output_wav_path,assistant_avatar,sentences)
                                else:
                                    st.error("生成に失敗したのだ",icon="✖")
                                    st.stop()

                    except Exception as e:
                        st.error(f"推論中にエラーが発生したのだ: {e}",icon="✖")
                        st.stop()


# 新しいチャットボタン
with new_chat_button:
    if st.button(label="新しいチャット",icon="🆕"):
        st.session_state["messages"] = []
        main_container.success("チャット履歴をクリアして、新しいチャットにしたのだ",icon="✅")
        time.sleep(1)
        st.rerun()
