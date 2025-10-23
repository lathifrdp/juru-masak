import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# --- KONFIGURASI ---

# Inisialisasi Klien Gemini
# Klien akan otomatis mencari GEMINI_API_KEY dari environment variable
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY tidak ditemukan. Cek file .env Anda.")
        
    client = genai.Client(api_key=api_key)
except Exception as e:
    print("Gagal menginisialisasi klien. Pastikan GEMINI_API_KEY sudah diatur.")
    print(f"Error: {e}")
    exit()

# Model yang akan digunakan
MODEL = 'gemini-2.5-flash' 

# Tentukan persona, gaya, dan aturan agen
PERSONA_PROMPT = (
    "Anda adalah 'Juru Masak Cerdas', seorang koki AI yang sangat ramah, penuh semangat, dan ahli dalam masakan Asia Tenggara. "
    "Gaya bahasa Anda santai dan menggugah selera. Selalu tambahkan sedikit saran atau cerita unik tentang masakan yang ditanyakan. "
    "Setiap respons HARUS dimulai dengan sapaan yang bersemangat (misalnya, 'Wah, ide yang bagus!') dan diakhiri dengan ajakan untuk memasak."
    "Tugas utama Anda adalah memberikan resep atau tips memasak."
    "Bila konteksinya tidak terkait dengan memasak, jawab dengan sopan bahwa Anda hanya fokus pada masakan dan tawarkan untuk membantu dengan resep atau tips memasak."
)

PERSONA_PROMPT2 = (
    "Anda adalah 'Pemain Bola'"
    "Anda mengerti peraturan bola"
    "Jangan jawab bila pertanyaannya tidak ada hubungannya dengan bola"
)

# --- FUNGSI UTAMA AGENT ---

def jalankan_agent_masak():
    """Mengelola loop percakapan dengan 'Juru Masak Cerdas'."""
    
    print("--- AI Agent: Juru Masak Cerdas ---")
    print(f"Model yang digunakan: {MODEL}")
    print("Ketik 'keluar' untuk mengakhiri.")
    print("-" * 35)

    thinking_config = types.ThinkingConfig(
        thinking_budget=0  # Menonaktifkan proses penalaran (lebih hemat token)
    )

    # Konfigurasi system instruction untuk menyuntikkan persona
    config = types.GenerateContentConfig(
        system_instruction=PERSONA_PROMPT,
        thinking_config=thinking_config
    )

    # Inisialisasi riwayat chat
    chat = client.chats.create(
        model=MODEL,
        config=config
    )

    while True:
        # Minta input pengguna
        user_input = input("Anda: ")
        
        if user_input.lower() in ['keluar', 'exit', 'stop']:
            print("\n👋 Sampai jumpa! Selamat mencoba resep-resep baru!")
            break

        print("Juru Masak Cerdas (memproses)...")
        
        try:
            # Kirim prompt pengguna ke model melalui objek chat
            response = chat.send_message(user_input)
            
            # Tampilkan respons dari agen
            print(f"\nJuru Masak Cerdas: {response.text}\n")
            
        except Exception as e:
            print(f"\nTerjadi kesalahan: {e}. Coba lagi.")
            break

# Jalankan fungsi utama
if __name__ == "__main__":
    jalankan_agent_masak()