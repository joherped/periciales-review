from openai import OpenAI
import os, json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_DIR = "Master-Periciales"
OUTPUT_FILE = "embeddings.json"
embeddings_data = []

for fn in os.listdir(INPUT_DIR):
    if fn.lower().endswith(".txt"):
        path = os.path.join(INPUT_DIR, fn)
        text = open(path, "r", encoding="utf-8").read()
        # Nueva llamada a embeddings
        resp = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        # Aquí la corrección: usamos .data[0].embedding
        emb = resp.data[0].embedding
        embeddings_data.append({
            "id": os.path.splitext(fn)[0],
            "text": text,
            "embedding": emb
        })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(embeddings_data, f, ensure_ascii=False)

print(f"✅ Generados {len(embeddings_data)} embeddings en {OUTPUT_FILE}")