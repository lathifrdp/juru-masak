import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

# Model Embedding eksternal
EMBEDDING_MODEL_HF = 'naufalihsan/indonesian-sbert-large'
print(f"Memuat model Sentence Transformer: {EMBEDDING_MODEL_HF}...")
try:
    # Inisialisasi model SentenceTransformer
    hf_model = SentenceTransformer(EMBEDDING_MODEL_HF)
    print("Model Sentence Transformer berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model Sentence Transformer: {e}")
    exit()

# Data Contoh (Ini akan menjadi *Document* yang Anda *embed* ke Vector DB)
KNOWLEDGE_BASE = [
    ("Resep Nasi Goreng Kampung", "Bahan: 200g nasi, 2 siung bawang merah, 1 siung bawang putih, 5g terasi. Cara: Tumis bumbu halus lalu masukkan nasi."),
    ("Tentang Lengkuas", "Rimpang keras beraroma pinus. Tidak bisa dimakan langsung, hanya untuk aroma."),
    ("Tips Membuat Sambal Terasi", "Gunakan 10g terasi udang kualitas super untuk hasil maksimal.")
]

# 1. PRA-PROSES: Membuat Embedding untuk seluruh dokumen di Knowledge Base
# Dalam Vector DB riil, langkah ini dilakukan sekali saat mengindeks data.
# documents = [content for title, content in KNOWLEDGE_BASE]
document_embeddings = hf_model.encode(KNOWLEDGE_BASE) #convert_to_tensor=False
print(f"Berhasil membuat {len(document_embeddings)} embedding dokumen.")

# --- FUNCTION CALLING DEFINITION ---
def hitung_porsi(bahan_dasar: str, jumlah_porsi: int):
    """Menghitung estimasi kelipatan bahan untuk jumlah porsi tertentu."""
    return f"Untuk {jumlah_porsi} porsi {bahan_dasar}, bumbunya perlu dikalikan {jumlah_porsi} kali dari resep standar."

def konversi_satuan(jumlah: float, satuan_asal: str, satuan_tujuan: str):
    """
    Mengonversi satuan bahan makanan (misal: gram ke sendok makan).
    """
    # Logika sederhana: 1 sendok makan (sdm) asumsikan 15 gram
    if satuan_asal.lower() == "gram" and satuan_tujuan.lower() == "sendok makan":
        hasil = jumlah / 15
        return {"hasil": f"{hasil:.1f} sendok makan"}
    elif satuan_asal.lower() == "siung" and satuan_tujuan.lower() == "gram":
        hasil = jumlah * 5
        return {"hasil": f"{hasil} gram"}
    return {"hasil": "Konversi tidak tersedia, gunakan perkiraan saja."}

def cari_substitusi_bahan(nama_bahan: str):
    """
    Memberikan saran bahan pengganti jika bahan utama tidak tersedia.
    """
    substitusi = {
        "terasi": "Kecap asin atau miso paste (untuk rasa umami serupa).",
        "lengkuas": "Jahe (meskipun aromanya sedikit berbeda).",
        "bawang merah": "Bawang bombay cincang.",
        "cabai": "Saus sambal atau bubuk paprika."
    }
    saran = substitusi.get(nama_bahan.lower(), "Maaf, saya belum punya saran pengganti untuk bahan itu.")
    return {"bahan_pengganti": saran}

# Fungsi untuk mencari data relevan menggunakan kemiripan vektor
def find_relevant_documents_vector(query, knowledge_base, doc_embeddings, top_k=1):
    """
    Mencari data paling relevan dari knowledge base menggunakan cosine similarity.
    Ini adalah simulasi dari apa yang dilakukan oleh Vector Database.
    """
    
    # 2. EMBEDDING QUERY: Buat embedding untuk pertanyaan pengguna
    query_embedding = hf_model.encode([query]) #convert_to_tensor=False
    
    # 3. PENCARIAN (Vector Search): Hitung kemiripan vektor
    # Menggunakan cosine similarity untuk mengukur kemiripan antara query dan dokumen
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    print(f"Skor kemiripan: {similarities}")
    
    # 4. RANKING: Dapatkan indeks dokumen yang paling mirip
    # np.argsort mengurutkan dari kecil ke besar, [::-1] membalik (dari besar ke kecil)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 5. RETRIEVAL: Ambil dokumen berdasarkan indeks
    relevant_docs = []
    for i in top_indices:
        # Hanya ambil jika kemiripan di atas batas tertentu (misalnya, 0.5)
        if similarities[i] > 0.5: 
            title, content = knowledge_base[i]
            relevant_docs.append(f"### Judul: {title} (Skor: {similarities[i]:.3f})\n### Konten: {content}")
            print(f"ambil data dengan similiarity: , {similarities[i]:.3f}, \n### Konten: {content}")
        else:
            # Jika skor kemiripan terlalu rendah, hentikan pencarian
            title, content = knowledge_base[i]
            print("Tidak similiar")
            print(f"### Judul: {title} (Skor: {similarities[i]:.3f})\n")
            break 
            
    return "\n---\n".join(relevant_docs) if relevant_docs else "Tidak ada informasi resep yang relevan"

# Tentukan persona, gaya, dan aturan agen
PERSONA_PROMPT = (
    "Anda adalah 'Juru Masak Cerdas', seorang koki AI yang sangat ramah, penuh semangat, dan ahli dalam masakan Asia Tenggara. "
    "Gaya bahasa Anda santai dan menggugah selera. Selalu tambahkan sedikit saran atau cerita unik tentang masakan yang ditanyakan. "
    "Setiap respons HARUS dimulai dengan sapaan yang bersemangat (misalnya, 'Wah, ide yang bagus!') dan diakhiri dengan ajakan untuk memasak."
    "Tugas utama Anda adalah memberikan resep atau tips memasak."
    "Sangat penting: JAWAB HANYA BERDASARKAN CONTEXTUAL DATA yang disajikan dalam prompt."
    "Jika informasi yang dibutuhkan tidak ada dalam CONTEXTUAL DATA, jawab dengan sopan bahwa Anda tidak memiliki resep atau tips terkait, dan alihkan ke topik masakan yang ada dalam database Anda."
    "Bila konteksinya tidak terkait dengan memasak, jawab dengan sopan bahwa Anda hanya fokus pada masakan dan tawarkan untuk membantu dengan resep atau tips memasak."
    "Mohon jawabnya dengan singkat saja"
    "gunakan fungsi 'hitung_porsi', 'konversi_satuan', 'cari_substitusi_bahan' untuk membantu."
)

# PERSONA_PROMPT2 = (
#     "Anda adalah 'Pemain Bola'"
#     "Anda mengerti peraturan bola"
#     "Jangan jawab bila pertanyaannya tidak ada hubungannya dengan bola"
# )

# --- FUNGSI UTAMA AGENT ---

def jalankan_agent_masak():
    """Mengelola loop percakapan dengan 'Juru Masak Cerdas'."""
    
    print("--- AI Agent: Juru Masak Cerdas (Mode RAG Vektor Aktif) ---")
    print(f"Model Gen: {MODEL} | Model Emb: {EMBEDDING_MODEL_HF}")
    print("Ketik 'keluar' untuk mengakhiri.")
    print("-" * 35)

    thinking_config = types.ThinkingConfig(
        thinking_budget=0  # Menonaktifkan proses penalaran (lebih hemat token)
    )

    # Konfigurasi system instruction untuk menyuntikkan persona
    config = types.GenerateContentConfig(
        system_instruction=PERSONA_PROMPT,
        thinking_config=thinking_config,
        max_output_tokens=100,
        tools=[hitung_porsi, konversi_satuan, cari_substitusi_bahan], # Menambahkan fungsi ke model
        #automatic_function_calling=types.AutomaticFunctionCallingConfig(max_remote_calls=3)
        automatic_function_calling=types.AutomaticFunctionCallingConfig()
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
            # 1. LANGKAH RAG: Cari dokumen paling relevan menggunakan Vector Search
            konteks_rag = find_relevant_documents_vector(
                user_input, 
                KNOWLEDGE_BASE, 
                document_embeddings, 
                top_k=2 # Ambil 2 dokumen paling relevan
            )

            if(konteks_rag == "Tidak ada informasi resep yang relevan"):
                print("\nJuru Masak Cerdas: Maaf, saya tidak memiliki informasi terkait resep atau tips memasak berdasarkan pertanyaan Anda. Namun, saya senang membantu Anda dengan resep atau tips memasak lainnya! Apa yang ingin Anda coba masak hari ini?\n")
                continue
            
            # 2. LANGKAH RAG: Susun prompt dengan konteks yang diambil
            RAG_PROMPT = (
                f"Gunakan hanya informasi dari CONTEXTUAL DATA untuk menjawab. Jaga persona Anda sebagai 'Juru Masak Cerdas' (ramah, semangat, ahli masakan Asia Tenggara). "
                f"Jika informasi tidak ada di CONTEXTUAL DATA, katakan Anda tidak dapat menjawabnya. Jawab pertanyaan pengguna: '{user_input}'\n\n"
                f"### CONTEXTUAL DATA ###\n"
                f"{konteks_rag}\n"
                f"### PERTANYAAN PENGGUNA ###\n"
                f"{user_input}"
            )

            # print(RAG_PROMPT)
            # Kirim prompt pengguna ke model melalui objek chat
            response = chat.send_message(RAG_PROMPT)
            
            # Tampilkan respons dari agen
            print(f"\nJuru Masak Cerdas: {response.text}\n")
            
        except Exception as e:
            print(f"\nTerjadi kesalahan: {e}. Coba lagi.")
            break

# Jalankan fungsi utama
if __name__ == "__main__":
    jalankan_agent_masak()