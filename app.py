from flask import Flask, request, render_template
import joblib
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# โหลดโมเดล SVM ของคุณ
model = joblib.load('svm_model_accuracy_95.25.pkl')

def predict_image(image_bytes):
    # แปลงภาพที่ได้รับเป็นข้อมูลที่โมเดลต้องการ
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('L')  # แปลงเป็นภาพขาวดำ (ถ้าจำเป็น)
    image = image.resize((224, 224))  # ปรับขนาดภาพ
    image_array = np.array(image).flatten().reshape(1, -1)  # แปลงเป็น array และ reshape สำหรับ SVM
    
    prediction = model.predict(image_array)  # ใช้โมเดลในการทำนาย
    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html')  # หน้าเว็บที่ให้ผู้ใช้อัพโหลดภาพ

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        image_bytes = file.read()  # อ่านไฟล์เป็น bytes
        prediction = predict_image(image_bytes)  # ทำนายผล
        return f'Predicted Class: {prediction}'  # ส่งผลการทำนายกลับไปที่ผู้ใช้

if __name__ == '__main__':
    app.run(debug=True)
