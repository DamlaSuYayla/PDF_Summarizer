================== PDF SUMMARIZER - AI Assistant ====================
<img width="2398" height="1346" alt="Ekran görüntüsü 2026-02-16 123636" src="https://github.com/user-attachments/assets/d6b53f7c-81c2-4c44-b005-d05eb2db031d" />

An AI-powered PDF summarization tool using the LED (Longformer Encoder-Decoder)
model with a Gradio web interface.

--------------------------------------------------------------------------------
FEATURES
--------------------------------------------------------------------------------
  • Upload PDF files and get AI-generated summaries
  • Text-to-Speech conversion for generated summaries
  • Fine-tuning capability with your own PDF-summary pairs
  • Web interface accessible from any device

--------------------------------------------------------------------------------
INSTALLATION
--------------------------------------------------------------------------------
1. Install Python 3.10 or higher

2. Install dependencies:
   pip install -r requirements.txt

3. Run the application:
   python Backend.py

4. Open the URL shown in terminal (default: http://localhost:7861)

--------------------------------------------------------------------------------
PROJECT STRUCTURE
--------------------------------------------------------------------------------
  PdfSummarizer/
  ├── Backend.py          - Main application (Gradio UI + Summarizer)
  ├── train.py            - Fine-tuning script
  ├── requirements.txt    - Python dependencies
  ├── training_data/      - Training data folder
  │   ├── *.pdf           - PDF files for training
  │   └── *.txt           - Corresponding summaries
  ├── checkpoints/        - Saved model checkpoints
  ├── outputs/            - Generated audio files
  └── logs/               - Training logs (TensorBoard)

--------------------------------------------------------------------------------
FINE-TUNING THE MODEL
--------------------------------------------------------------------------------
1. Add your PDF files to the `training_data/` folder

2. For each PDF, create a .txt file with the SAME name containing the ideal
   summary (e.g., report.pdf -> report.txt)

3. Run the training script:
   python train.py

4. The fine-tuned model will be saved to `checkpoints/final_model/`

5. The application will automatically use the fine-tuned model if available

--------------------------------------------------------------------------------
CONFIGURATION
--------------------------------------------------------------------------------
  Model:      allenai/led-base-16384 (default)
  Max Tokens: 4096
  Device:     CUDA (GPU) if available, otherwise CPU

--------------------------------------------------------------------------------
METRICS (after fine-tuning)
--------------------------------------------------------------------------------
  • ROUGE-1    - Unigram overlap score
  • ROUGE-L    - Longest common subsequence score
  • BLEU       - Bilingual evaluation score
  • BERTScore  - Semantic similarity score

Metrics are saved to `metrics.json` after training.

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
  • Python 3.10+
  • 8GB+ RAM (16GB recommended for training)
  • GPU with CUDA support (optional, but recommended for faster processing)

================================================================================
