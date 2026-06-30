from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

print("Starting...")

SAVE_PATH   = "./best_model.pth"
DEVICE      = torch.device("cpu")
CLASS_NAMES = [
    "david_fincher", "denis_villeneuve", "guillermo_del_toro",
    "stanley_kubrick", "tim_burton", "wes_anderson"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Loading model...")
model = models.resnet18(weights=None)
model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, len(CLASS_NAMES)))
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
model.eval()
print("Model loaded.")

app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Director Style Classifier</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: #0f0f0f; color: #e8e8e8; min-height: 100vh;
           display: flex; align-items: center; justify-content: center; padding: 2rem; }
    .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px;
            padding: 2rem; width: 100%; max-width: 520px; }
    h1 { font-size: 1.3rem; font-weight: 600; margin-bottom: 0.4rem; }
    .subtitle { font-size: 0.875rem; color: #888; margin-bottom: 1.75rem; }
    .drop-zone { border: 2px dashed #333; border-radius: 8px; padding: 2.5rem 1rem;
                 text-align: center; cursor: pointer; transition: border-color 0.2s;
                 margin-bottom: 1.5rem; }
    .drop-zone:hover { border-color: #666; }
    .drop-zone p { font-size: 0.875rem; color: #666; }
    .drop-zone span { color: #aaa; font-weight: 500; }
    #preview { width: 100%; border-radius: 8px; margin-bottom: 1.5rem; display: none; }
    #file-input { display: none; }
    .result-label { font-size: 0.75rem; text-transform: uppercase;
                    letter-spacing: 0.08em; color: #555; margin-bottom: 1rem; }
    .bar-row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.6rem; }
    .bar-name { font-size: 0.8rem; width: 160px; flex-shrink: 0; color: #ccc; }
    .bar-name.top { color: #fff; font-weight: 600; }
    .bar-track { flex: 1; background: #2a2a2a; border-radius: 4px; height: 6px; overflow: hidden; }
    .bar-fill { height: 100%; background: #555; border-radius: 4px; transition: width 0.5s ease; }
    .bar-fill.top { background: #e8e8e8; }
    .bar-pct { font-size: 0.75rem; color: #666; width: 38px; text-align: right; flex-shrink: 0; }
    .bar-pct.top { color: #e8e8e8; }
    #results { display: none; }
    #loading { text-align: center; font-size: 0.875rem; color: #555; display: none; margin-bottom: 1rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Director Style Classifier</h1>
    <p class="subtitle">Upload a movie frame to identify the director's visual style.</p>
    <div class="drop-zone" id="drop-zone">
      <p><span>Click to upload</span> or drag and drop</p>
      <p>JPG, PNG supported</p>
    </div>
    <input type="file" id="file-input" accept="image/*">
    <img id="preview" alt="Uploaded frame">
    <p id="loading">Analysing...</p>
    <div id="results">
      <p class="result-label">Results</p>
      <div id="bars"></div>
    </div>
  </div>
  <script>
    const dropZone  = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const preview   = document.getElementById("preview");
    const results   = document.getElementById("results");
    const loading   = document.getElementById("loading");
    const bars      = document.getElementById("bars");
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.style.borderColor = "#666"; });
    dropZone.addEventListener("dragleave", () => { dropZone.style.borderColor = "#333"; });
    dropZone.addEventListener("drop", e => { e.preventDefault(); dropZone.style.borderColor = "#333"; handleFile(e.dataTransfer.files[0]); });
    fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));
    function handleFile(file) {
      if (!file) return;
      preview.src = URL.createObjectURL(file);
      preview.style.display = "block";
      results.style.display = "none";
      loading.style.display = "block";
      bars.innerHTML = "";
      const formData = new FormData();
      formData.append("image", file);
      fetch("/predict", { method: "POST", body: formData })
        .then(r => r.json())
        .then(data => {
          loading.style.display = "none";
          results.style.display = "block";
          data.forEach((item, idx) => {
            const isTop = idx === 0;
            bars.innerHTML += `<div class="bar-row">
              <span class="bar-name ${isTop ? "top" : ""}">${item.name}</span>
              <div class="bar-track"><div class="bar-fill ${isTop ? "top" : ""}" style="width:${item.confidence}%"></div></div>
              <span class="bar-pct ${isTop ? "top" : ""}">${item.confidence}%</span>
            </div>`;
          });
        });
    }
  </script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    file   = request.files["image"]
    img    = Image.open(io.BytesIO(file.read())).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().tolist()
    results = [{"name": CLASS_NAMES[i].replace("_", " ").title(),
                "confidence": round(probs[i] * 100, 1)}
               for i in range(len(CLASS_NAMES))]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)