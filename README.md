# Serenity AI: Private Mental Health Support Assistant

Serenity AI is an end-to-end, privacy-focused mental health support assistant designed to provide empathetic, context-grounded responses. The project combines a custom fine-tuned Large Language Model (LLM) with a Retrieval-Augmented Generation (RAG) pipeline to deliver factually backed information from clinical guidelines while ensuring user data privacy through fully local deployment.

## 🚀 Key Features

* **Empathetic Fine-Tuning:** A base model optimized using QLoRA to deliver supportive conversational interactions tailored for mental health contexts.
* **Hybrid RAG Architecture:** Integrates real-time retrieval from official WHO and NIMH resources to reduce hallucinations and ground responses in verifiable information.
* **Privacy-First Design:** All inference, embeddings, and vector storage operate locally without transmitting user conversations to external services.
* **Hardware Optimized:** Quantized to GGUF format for low-latency execution on consumer hardware (tested on an NVIDIA RTX 3050).
* **Transparent Responses:** Streamlit interface with clinical source citations for improved trust and explainability.

---

## 🏗️ System Architecture

The project consists of two core pipelines:

### 1. Knowledge Base Builder (`ingest.py`)

* Loads clinical documents from PDFs and CSV files
* Splits content into semantic chunks
* Generates vector embeddings
* Persists the vector database locally using ChromaDB

### 2. Inference Engine & Interface (`app.py`)

* Provides a conversational Streamlit interface
* Performs semantic similarity searches against the vector store
* Injects retrieved clinical context into the fine-tuned LLM
* Displays responses with supporting source citations

---

## 📦 Tech Stack

| Component            | Technology                       |
| -------------------- | -------------------------------- |
| **LLM Framework**    | Unsloth, Llama 3.2 (3B Instruct) |
| **Fine-Tuning**      | QLoRA                            |
| **RAG Pipeline**     | LangChain, ChromaDB              |
| **Embeddings**       | Hugging Face `all-MiniLM-L6-v2`  |
| **Inference Format** | GGUF (Q4_K_M Quantization)       |
| **Frontend**         | Streamlit with Custom CSS        |

---

## 🤖 Model Weights

To keep the repository lightweight, the compiled model weights are hosted externally.

* **Fine-Tuned GGUF Weights:** [Hugging Face Model Repository](https://huggingface.co/your-username/your-model)

> **Note:** The quantized model file is approximately 3 GB in size.

---

## 📂 Project Structure

```text
Serenity-AI/
├── app.py
├── ingest.py
├── data/
│   ├── who_guidelines.pdf
│   └── nimh_resources.csv
├── models/
│   └── model_final.gguf
├── vectorstore/
├── requirements.txt
└── README.md
```

---

## 🔧 Installation & Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/repository-name.git
cd repository-name
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Build the Knowledge Base

Place your reference documents (PDFs or CSV files containing WHO/NIMH guidelines) inside the `./data` directory.

Run the ingestion pipeline:

```bash
python ingest.py
```

This process will generate and persist the local vector database.

### 5. Download Model Weights

Create a `models` directory in the project root:

```bash
mkdir models
```

Download the `model_final.gguf` file from the Hugging Face repository and place it inside:

```text
./models/model_final.gguf
```

### 6. Launch the Application

Start the Streamlit interface:

```bash
streamlit run app.py
```

Open your browser and navigate to:

```text
http://localhost:8501
```

---

## 📊 Fine-Tuning Specifications

| Parameter           | Value                                  |
| ------------------- | -------------------------------------- |
| **Dataset**         | `nbertagnolli/counsel-chat`            |
| **Methodology**     | QLoRA via Unsloth                      |
| **LoRA Rank (`r`)** | 16                                     |
| **LoRA Alpha**      | 16                                     |
| **Learning Rate**   | `2e-4`                                 |
| **Target Modules**  | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

### Optimization Techniques

* 4-bit quantization for efficient training and inference
* Smart gradient offloading to reduce VRAM usage
* GGUF conversion for lightweight local deployment

---

## 🔒 Privacy & Security

Serenity AI is designed with privacy as a core principle.

* All model inference occurs locally
* User conversations are not transmitted to external APIs
* Clinical documents remain on the user's device
* Vector embeddings and databases are stored locally

---

## ⚠️ Limitations & Ethical Disclaimer

### Non-Clinical Tool

Serenity AI is intended exclusively for informational and supportive purposes. It is **not** a substitute for professional mental health care and does not provide:

* Clinical diagnoses
* Medical treatment recommendations
* Crisis intervention services
* Pharmaceutical prescriptions

### Seek Professional Support

If you or someone you know is experiencing severe emotional distress, thoughts of self-harm, or a mental health crisis, please contact local emergency services or seek support from a licensed mental health professional immediately.
