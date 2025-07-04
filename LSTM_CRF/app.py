from flask import Flask, request, jsonify
from predict import model2predict

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def index():
    try:
        data = request.get_json()
        print("解析后的 JSON 数据:", data)
        if not data or 'text' not in data:
            return jsonify({"error": "请输入有效的 JSON 数据，包含 'text' 字段"}), 400

        result = model2predict(data)
        return jsonify({"text": data['text'], "predict": result})
    except Exception as e:
        print("JSON 解析或处理错误:", str(e))
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)
