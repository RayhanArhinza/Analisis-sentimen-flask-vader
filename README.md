# README

## 📌 Prerequisites
Pastikan Anda telah menginstall Python sebelum melanjutkan. Anda dapat mengunduhnya di [python.org](https://www.python.org/downloads/).

## 🚀 Instalasi
1. **Buat Virtual Environment (Opsional tetapi Disarankan)**
   ```sh
   python -m venv env
   source env/bin/activate  # MacOS/Linux
   env\Scripts\activate  # Windows
   ```

2. **Perbarui pip**
   ```sh
   pip install --upgrade pip
   ```

3. **Install Dependensi yang Dibutuhkan**
   ```sh
   pip install torch nltk flask pandas numpy matplotlib transformers
   ```

4. **Download Resource NLTK**
   ```sh
   python -m nltk.downloader punkt stopwords
   ```

## 📂 Struktur Direktori
```
project-folder/
│── app.py  # Main Flask Application
│── templates/
│   ├── index.html  # Frontend template
│── static/
│   ├── style.css  # Optional CSS
│── README.md  # Panduan Instalasi
```

## 🎯 Cara Menjalankan Aplikasi
1. Jalankan perintah berikut untuk menjalankan aplikasi Flask:
   ```sh
   python app.py
   ```
2. Buka browser dan akses:
   ```
   http://127.0.0.1:5000/
   ```

## 🛠 Troubleshooting
Jika terjadi error, coba jalankan perintah berikut:
- **Memastikan Python Terinstall:**
  ```sh
  python --version
  ```
- **Memeriksa Dependensi yang Terinstall:**
  ```sh
  pip list
  ```
- **Menghapus dan Menginstall Ulang Dependensi:**
  ```sh
  pip freeze > requirements.txt
  pip uninstall -r requirements.txt -y
  pip install -r requirements.txt
  ```
=

---
🔥 Selamat coding! Jika ada pertanyaan, jangan ragu untuk menghubungi saya! 😊

