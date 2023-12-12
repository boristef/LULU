from flask import Flask, render_template, request
import jieba
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

app = Flask(__name__)


def chinese_text_segmentation(text):
    seg_list = jieba.cut(text)
    return " ".join(seg_list)


max_words = 1000
max_len = 20

# 載入模型
model = load_model("LSTMmodel_20231208.h5")  # 請替換成實際的模型路徑
tokenizer = Tokenizer(num_words=max_words)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        user_input_segmented = chinese_text_segmentation(user_input)
        user_input_seq = tokenizer.texts_to_sequences([user_input_segmented])
        user_input_pad = pad_sequences(user_input_seq, maxlen=max_len)
        predictions = model.predict(user_input_pad)
        predicted_class = predictions.argmax(axis=-1)[0]
        return render_template(
            "index.html", user_input=user_input, predicted_class=predicted_class
        )
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
