import os, re, gc, logging, warnings
from pathlib import Path
from datetime import datetime
from typing import List, Optional

warnings.filterwarnings("ignore")

# Optional imports — simplified
def _try_import(name, pkg=None):
    try:
        return __import__(name), True
    except Exception:
        logging.warning(f"{pkg or name} not installed; related features will be unavailable.")
        return None, False

torch, TORCH_AVAILABLE = _try_import('torch')
fitz, FITZ_AVAILABLE = _try_import('fitz', pkg='PyMuPDF')
nltk, NLTK_AVAILABLE = _try_import('nltk')

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"

gr, GRADIO_AVAILABLE = _try_import('gradio')

# Simplified gTTS import using helper
_gtts_mod, GTTS_AVAILABLE = _try_import('gtts')
if _gtts_mod:
    gTTS = getattr(_gtts_mod, 'gTTS', None)
else:
    gTTS = None

# Simplified transformers import using helper
_trans_mod, TRANSFORMERS_AVAILABLE = _try_import('transformers')
if _trans_mod:
    AutoTokenizer = getattr(_trans_mod, 'AutoTokenizer', None)
    LEDForConditionalGeneration = getattr(_trans_mod, 'LEDForConditionalGeneration', None)
else:
    AutoTokenizer = None
    LEDForConditionalGeneration = None

# NLTK setup
try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt', quiet=True)

# Fallback sentence tokenizer with minimal external deps
def sent_tokenize(text: str):
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        try:
            nltk.download('punkt', quiet=True)
            return nltk.sent_tokenize(text)
        except Exception:
            # Fallback naive splitter
            return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


# Config
DEVICE = "cuda" if (TORCH_AVAILABLE and getattr(torch, 'cuda', None) is not None and torch.cuda.is_available()) else "cpu"
if not TORCH_AVAILABLE:
    logging.info("PyTorch not available; running on CPU. Install torch to enable model inference.")

MODEL_NAME = "allenai/led-base-16384"
MAX_TOKENS = 4096
OVERLAP = 0.15

# Get base directory (where Backend.py is located)
BASE_DIR = Path(__file__).parent.resolve()

# Directories (use absolute paths)
for d in ["checkpoints", "training_data", "outputs", "logs"]:
    (BASE_DIR / d).mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== DATA CLEANING ====================
class DataCleaner:
    PATTERNS = {
        'refs': [r'\[[\d,\s-]+\]', r'\([\w\s]+,\s*\d{4}\)', r'(?:References|Bibliography)[\s\S]*$'],
        'tables': [r'(?:Table)\s+\d+[.:][^\n]*', r'[-|+]{3,}'],
        'figures': [r'(?:Figure|Fig)\s+\d+[.:][^\n]*'],
        'footnotes': [r'^\s*\d+\s+[A-Z].*$', r'^\s*\*+\s*.+$'],
        'code': [r'```[\s\S]*?```', r'(?:def|class|import)\s+\w+']
    }
    
    def clean(self, text: str) -> str:
        if not text: return ""
        for patterns in self.PATTERNS.values():
            for p in patterns:
                text = re.sub(p, '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        sentences = sent_tokenize(text)
        return ' '.join(s for s in sentences if len(s.split()) >= 5)

# ==================== CHUNKING ====================
class Chunker:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
    
    def chunk(self, text: str) -> List[str]:
        if not self.tokenizer: return [text]
        sentences = sent_tokenize(text)
        if not sentences: return []
        
        chunks, current = [], [sentences[0]]
        for i in range(1, len(sentences)):
            chunk_text = ' '.join(current + [sentences[i]])
            tokens = len(self.tokenizer.encode(chunk_text, add_special_tokens=False))
            if tokens <= MAX_TOKENS:
                current.append(sentences[i])
            else:
                chunks.append(' '.join(current))
                current = [sentences[i]]
        if current: chunks.append(' '.join(current))
        
        if len(chunks) > 1:
            for i in range(1, len(chunks)):
                prev_words = chunks[i-1].split()
                overlap_words = prev_words[-int(len(prev_words)*OVERLAP):]
                chunks[i] = ' '.join(overlap_words) + ' ' + chunks[i]
        return chunks

# ==================== SUMMARIZER ====================
class Summarizer:
    def __init__(self, model_path=MODEL_NAME):
        self.model_path = model_path
        self.model = self.tokenizer = None
        self.cleaner = DataCleaner()
        self.chunker = None  # Will be initialized after tokenizer loads
        self.max_chunks = 20
        self.max_depth = 3
    
    def _load(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = LEDForConditionalGeneration.from_pretrained(self.model_path).to(DEVICE).eval()
            self.chunker = Chunker(self.tokenizer)  # Share tokenizer
    
    def _clean_summary(self, text: str) -> str:
        text = re.sub(r'\[\d+[\d,\s-]*\]', '', text)
        text = re.sub(r'\([A-Za-z\s]+,?\s*\d{4}\)', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove leading punctuation
        text = re.sub(r'^[.,;:!?\-–—\s]+', '', text)
        if text and not text.endswith(('.', '!', '?')):
            sentences = text.rsplit('.', 1)
            if len(sentences) > 1:
                text = sentences[0] + '.'
        return text
    
    def _summarize_chunk(self, text: str) -> str:
        self._load()
        inputs = self.tokenizer(text, max_length=MAX_TOKENS, truncation=True, return_tensors="pt").to(DEVICE)
        
        # LED modeli için global attention mask gerekli
        global_attention_mask = torch.zeros_like(inputs['input_ids'])
        global_attention_mask[:, 0] = 1  # İlk token'a global attention
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                global_attention_mask=global_attention_mask,
                max_length=400,
                min_length=50,
                num_beams=2,
                length_penalty=1.5,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        return self._clean_summary(self.tokenizer.decode(out[0], skip_special_tokens=True))
    
    def summarize(self, text: str, depth: int = 0) -> str:
        text = self.cleaner.clean(text)
        if not text: return "Text could not be extracted."
        
        self._load()
        tokens = len(self.tokenizer.encode(text))
        
        if tokens <= MAX_TOKENS:
            return self._summarize_chunk(text)
        
        if depth >= self.max_depth:
            return self._summarize_chunk(text[:MAX_TOKENS * 4])
        
        chunks = self.chunker.chunk(text)
        if len(chunks) > self.max_chunks:
            step = len(chunks) // self.max_chunks
            chunks = chunks[::step][:self.max_chunks]
        
        print(f"📝 Summarizing {len(chunks)} chunks...")
        summaries = [self._summarize_chunk(c) for c in chunks]
        combined = ' '.join(summaries)
        
        if len(self.tokenizer.encode(combined)) > MAX_TOKENS:
            return self.summarize(combined, depth + 1)
        return self._summarize_chunk(combined)

# ==================== PDF & TTS ====================
def extract_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = '\n'.join(p.get_text() for p in doc)
    doc.close()
    return text

def text_to_speech(text: str) -> Optional[str]:
    if not text: return None
    path = str(BASE_DIR / "outputs" / f"audio_{datetime.now():%Y%m%d_%H%M%S}.mp3")
    gTTS(text=text, lang='en').save(path)
    return path

# ==================== GRADIO UI ====================
fine_tuned_path = BASE_DIR / "checkpoints" / "final_model"
model_path = str(fine_tuned_path) if fine_tuned_path.exists() else MODEL_NAME
summarizer = Summarizer(model_path)

def process_pdf(pdf_file):
    if not pdf_file: return "📤 Please upload a PDF file first"
    text = extract_pdf(pdf_file.name)
    return summarizer.summarize(text)

def get_audio(text):
    if not text or text.startswith("📤"): return None, None
    path = text_to_speech(text)
    return path, path

# UI with Gradio's built-in Ocean theme
with gr.Blocks(title="PDF Summarizer", theme=gr.themes.Ocean()) as app:
    gr.Markdown("# 📄 PDF Summarizer")
    gr.Markdown("*Transform lengthy documents into concise summaries with AI*")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="📤 Upload PDF", file_types=[".pdf"])
            summarize_btn = gr.Button("✨ Generate Summary", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            summary_output = gr.Textbox(label="📝 Summary", lines=12, placeholder="Your summary will appear here...")
    
    gr.Markdown("### 🔊 Text-to-Speech")
    with gr.Row():
        audio_btn = gr.Button("🎧 Convert to Audio")
        audio_output = gr.Audio(label="Listen", type="filepath")
        download_output = gr.File(label="Download MP3")
    
    summarize_btn.click(fn=process_pdf, inputs=[pdf_input], outputs=[summary_output])
    audio_btn.click(fn=get_audio, inputs=[summary_output], outputs=[audio_output, download_output])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7861, share=True)
