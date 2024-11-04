from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Dictionary untuk mapping indeks kelas ke nama kelas
dic = {0: 'dyed-lifted-polyps', 1: 'dyed-resection-margins', 2: 'esophagitis', 3: 'normal-cecum', 
       4: 'normal-pylorus', 5: 'normal-z-line', 6: 'polyps', 7: 'ulcerative-colitis'}

# Muat model
model = load_model('hasil_ensemble.keras')
model.make_predict_function()

# Fungsi untuk memprediksi label gambar
def predict_label(img_path):
    i = image.load_img(img_path, target_size=(416,416))
    i = image.img_to_array(i)
    i = np.expand_dims(i, axis=0)
    # Tanpa normalisasi
    p = model.predict(i)

    # --- Tambahkan bagian ini ---
    # Tentukan threshold confidence score
    threshold = 0.7  # Contoh nilai threshold, sesuaikan dengan kebutuhan

    # Ambil confidence score tertinggi
    confidence = np.max(p)

    # Jika confidence score di bawah threshold, prediksi "Kelas tidak ditemukan"
    if confidence < threshold:
        return "Kelas tidak ditemukan"
    # --- Akhir bagian tambahan ---

    predicted_class = np.argmax(p, axis=1)[0]
    return dic[predicted_class]

# Route untuk halaman utama
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("classification.html")

# Route untuk memproses gambar yang diupload
@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename 
        img.save(img_path)

        p = predict_label(img_path)

    return render_template("classification.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
    app.run(debug = True)