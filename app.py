from flask import Flask, render_template, request
import jieba
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# 載入模型
model = load_model("LSTMmodel_20231208.h5")  # 請替換成實際的模型路徑

# 設定參數
max_words = 1000
max_len = 20

# 使用Tokenizer初始化
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts([""])  # 空列表即可


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["text_input"]
        user_input_segmented = chinese_text_segmentation(user_input)
        user_input_seq = tokenizer.texts_to_sequences([user_input_segmented])
        user_input_pad = pad_sequences(user_input_seq, maxlen=max_len)
        predictions = model.predict(user_input_pad)
        predicted_class = predictions.argmax(axis=-1)[0]
        return render_template("index.html", result=predicted_class)

    return render_template("index.html", result=None)


def chinese_text_segmentation(text):
    seg_list = jieba.cut(text)
    return " ".join(seg_list)


if __name__ == "__main__":
    app.run(debug=True)
