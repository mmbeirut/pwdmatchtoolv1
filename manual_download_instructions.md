# Manual Model Download Instructions

If the automatic download fails due to SSL issues, you can manually download the sentence transformer model:

## Step 1: Download Model Files
1. Go to: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
2. Click the "Files and versions" tab
3. Download the following files:
   - `config.json`
   - `pytorch_model.bin` 
   - `tokenizer.json`
   - `tokenizer_config.json`
   - `vocab.txt`
   - `modules.json`
   - `sentence_bert_config.json`

## Step 2: Create Local Model Directory
Create a folder called `local_model` in your project directory:
```
PWDMatchToolv5/
├── local_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   ├── modules.json
│   └── sentence_bert_config.json
├── app.py
├── requirements.txt
└── ...
```

## Step 3: Test the Model
Run the test script to verify the model works:
```bash
python test_model.py
```

## Step 4: Run the Application
If the test passes, run the main application:
```bash
python app.py
```

The application will automatically detect and use the local model.

## Alternative: Use Git LFS
If you have Git LFS installed, you can clone the model repository:
```bash
git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 local_model
```

## Troubleshooting
- Make sure all files are in the `local_model` directory
- Check that file sizes match those on Hugging Face (pytorch_model.bin should be ~90MB)
- Ensure you have sufficient disk space (~200MB total)
