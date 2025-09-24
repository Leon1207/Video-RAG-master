# Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension

[![Arxiv](https://img.shields.io/badge/Arxiv-2411.13093-red)](https://arxiv.org/abs/2411.13093)
![](https://img.shields.io/badge/Task-VideoQA-blue) [![Arxiv](https://img.shields.io/badge/Web-Project_Page-yellow)](https://video-rag.github.io/)
[![YouTube](https://img.shields.io/badge/-YouTube-000000?logo=youtube&logoColor=FF0000)](https://www.youtube.com/watch?v=WTs3xHicR_0)
[![Blog](https://img.shields.io/badge/Blog-LearnOpenCV-blue?style=flat&logo=data:image/png;base64,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)](https://learnopencv.com/video-rag-for-long-videos/)

<font size=7><div align='center' > [[🍎 Project Page](https://video-rag.github.io/)] [[📖 arXiv Paper](https://arxiv.org/pdf/2411.13093)]  </div></font>

## 😮 Highlights
![radar](https://github.com/user-attachments/assets/3ee6d1a7-24d7-42ff-a592-081491862ce8)

- **We integrate RAG into open-source LVLMs:** Video-RAG incorporates three types of visually-aligned auxiliary texts (OCR, ASR, and object detection) processed by external tools and retrieved via RAG, enhancing the LVLM. It’s implemented using completely open-source tools, without the need for any commercial APIs.
- **We design a versatile plug-and-play RAG-based pipeline for any LVLM:** Video-RAG offers a training-free solution for a wide range of LVLMs, delivering performance improvements with minimal additional resource requirements.
- **We achieve proprietary-level performance with open-source models:** Applying Video-RAG to a 72B open-source model yields state-of-the-art performance in Video-MME, surpassing models such as Gemini-1.5-Pro.
![framework](https://github.com/user-attachments/assets/9c9b176c-10a8-483e-be6b-de72b2b68191)
![results](https://github.com/user-attachments/assets/31ff86fa-9f00-44ae-bf44-9f164222dd7c)


## 🔨 Usage

This repo is built upon LLaVA-NeXT:

- Step 1: Clone and build LLaVA-NeXT conda environment:

```
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```
Then install the following packages in llava environment:
```
pip install spacy faiss-cpu easyocr ffmpeg-python
pip install torch==2.1.2 torchaudio numpy
python -m spacy download en_core_web_sm
# Optional: pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
```

- Step 2: Clone and build another conda environment for APE by: 

```
git clone https://github.com/shenyunhang/APE
cd APE
pip3 install -r requirements.txt
python3 -m pip install -e .
```

- Step 3: Copy all the files in `vidrag_pipeline` under the root dir of LLaVA-NeXT;

- Step 4: Copy all the files in `ape_tools` under the `demo` dir of APE;

- Step 5: Opening a service of APE by running the code under `APE/demo`:

```
python demo/ape_service.py
```

- Step 6: You can now run our pipeline build upon LLaVA-Video-7B by:

```
python vidrag_pipeline.py
```

> [!NOTE]
> You can also use our pipeline in any LVLMs by implementing some modifications in `vidrag_pipeline.py`:
```
1. The video-language model you load (line #161).
2. The llava_inference() function, make sure your model supports both inputs with/without video (line #175).
3. The process_video() function may suit your model (line #34).
4. The final prompt may suit your model (line #366).
```

## ✏️ Citation

If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝:

```
@misc{luo2024videoragvisuallyalignedretrievalaugmentedlong,
      title={Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension}, 
      author={Yongdong Luo and Xiawu Zheng and Xiao Yang and Guilin Li and Haojia Lin and Jinfa Huang and Jiayi Ji and Fei Chao and Jiebo Luo and Rongrong Ji},
      year={2024},
      eprint={2411.13093},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.13093}, 
}
```






