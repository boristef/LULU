from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import jieba
import pickle

app = Flask(__name__)

# loading
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
# 載入事先訓練好的模型
model = load_model("NNmodel_20231213.keras")  # 請替換成實際的模型路徑


def chinese_text_segmentation(text):
    seg_list = jieba.cut(text)
    print("seg_list:", seg_list)
    return " ".join(seg_list)


def convert_to_chinese(number):
    chinese_numerals = {
        "0": "零",
        "1": "一",
        "2": "二",
        "3": "三",
        "4": "四",
        "5": "五",
        "6": "六",
        "7": "七",
        "8": "八",
        "9": "九",
    }
    chinese_number = "".join([chinese_numerals[digit] for digit in str(number)])
    return chinese_number


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form["text"]
    print(user_input)
    user_input = "".join(
        [convert_to_chinese(ch) if ch.isdigit() else ch for ch in str(user_input)]
    )
    # 進行文本分詞並轉換成序列
    user_input_segmented = chinese_text_segmentation(user_input)
    print(user_input_segmented)
    user_input_seq = tokenizer.texts_to_sequences([user_input_segmented])
    print(user_input_seq)
    user_input_pad = pad_sequences(user_input_seq, maxlen=10)
    print(user_input_pad)

    # 預測結果
    predictions = model.predict(user_input_pad)
    predicted_class = predictions.argmax(axis=-1)[0]

    return jsonify(
        {"class": int(predicted_class), "probabilities": predictions.tolist()}
    )


if __name__ == "__main__":
    app.run(debug=True)
