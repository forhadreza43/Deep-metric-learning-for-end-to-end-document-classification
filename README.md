
1. ##Clone the repository

   ```bash
https://github.com/forhadreza43/Deep-metric-learning-for-end-to-end-document-classification.git
   ```

2. ## 📁 Project Structure

```
Project/
  rvl-cdip
  rvl-cdip-o
  rvl-cdip-o-text
  QS-OCR-Large
  test.txt
  train.txt
  val.txt
  instruction.txt
  src/
    config.py
    data.py
    sampler.py
    model.py
    loss.py
    knn_ood.py
    metrics.py
    train.py
    evaluate.py
```

Add all the datasets along with project structure.


3. ##Install dependencies and packages

   ```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets tokenizers
pip install faiss-cpu
pip install faiss-gpu
pip install numpy scikit-learn tqdm
   ```

4. ##Run the following command

   ```bash
   npm run dev
   ```

Train (debug)
   ```bash
   python src/train.py --project_root "D:\Thesis\Project" --save_dir checkpoints
   ```


Evaluate (KNN, no agreement)
   ```bash
   python src/evaluate.py --project_root "D:\Thesis\Project" --ckpt checkpoints/bert_margin_star_debug.pt
   ```

Evaluate (KNN* with consensus agreement)
   ```bash
   python src/evaluate.py --project_root "D:\Thesis\Project" --ckpt checkpoints\bert_margin_star_debug.pt --use_knn_star
   ```
