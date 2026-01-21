# Working Samples of To Eun Kim

## 1. Efficient LLM and RAG

### 1-1. Evaluation of Mamba State Space Model (SSM) ([Private Repo](https://github.com/kimdanny/kimdanny.github.io/tree/master/working_samples/bias-bench))
- Done in CMU Ethics course.
- [Report G-Drive Link](https://drive.google.com/file/d/1QN-aUDoGz9d3rjb1t4jJrwfnmbHkaOgR/view?usp=share_link)
- Implemented with **PyTorch**, **PEFT**, and **transformers**
- State Space Model (Mamba) Finetuning ([code](https://github.com/kimdanny/kimdanny.github.io/blob/master/working_samples/bias-bench/finetuning/finetune_mamba.py))
- Pythia Finetuning with LoRA (with PEFT) and GPU optimization ([code1-sft](https://github.com/kimdanny/kimdanny.github.io/blob/master/working_samples/bias-bench/finetuning/finetune_pythia.py)) ([code2-instruction-tuning](https://github.com/kimdanny/kimdanny.github.io/blob/master/working_samples/bias-bench/finetuning/instruction_tune_pythia.py))

### 1-2. Learning To Rank Retrievers (LTRR) for Efficient RAG ([Public Repo](https://github.com/kimdanny/Starlight-LiveRAG))
- [SIGIR LiveRAG Paper](https://arxiv.org/abs/2506.13743) (Spotlight presentation)
- Stochastic Reranker implementation ([code](https://github.com/kimdanny/Starlight-LiveRAG/blob/main/retrievers/ReRankers.py#L63))
- Score Regularization-based Reranker implementation ([code](https://github.com/kimdanny/Starlight-LiveRAG/blob/main/retrievers/ReRankers.py#L103))
- Point, Pair, List-wise Learning-To-Rank-Retriever implementation and training ([code](https://github.com/kimdanny/Starlight-LiveRAG/blob/main/04_train_ltrr.py))
    - **PyTorch** for data loading and LLM-based retriever-router training

### 1-3. Mixture of Retrievers (MoR) ([Public Repo](https://github.com/Josh1108/MixtureRetrievers))
- [EMNLP Paper](https://aclanthology.org/2025.emnlp-main.601/) (Main)



## 2. Implementing stuff from scratch
### 2-1. Llama from scratch (CMU ANLP course project)
- [private code](https://github.com/kimdanny/kimdanny.github.io/tree/master/working_samples/llama-from-scratch)

### 2-2. Full RAG pipeline (CMU ANLP course project)
- [private code](https://github.com/kimdanny/kimdanny.github.io/tree/master/working_samples/rag-pipeline)

### 2-3. Search Engine from scratch (CMU Search Engines course project)
- (part of the course) BERT Reranker implementation and reranking depth experiments ([report-1](https://drive.google.com/file/d/1Zs8-TxQl8J_1cEXLFTxHRx75_RQ0rIBe/view?usp=share_link)) ([report-2](https://drive.google.com/file/d/1iyXq3PoUa1rRg2QAN3H5yLpZBDuDeeAC/view?usp=share_link))




## 3. Misc.
### 3-1. Advertisement in Conversational Search ([Public Repo](https://github.com/kimdanny/TeamCMU-AdRAG))
- Qwen Ad-Rewriter SFT with TRL and DeepSpeed library ([code](https://github.com/kimdanny/TeamCMU-AdRAG/blob/main/ad_rewriter/train_sft_rewritier/train.py))
- DeBERTa-based Ad-Classifier training with curriculum learning ([code](https://github.com/kimdanny/TeamCMU-AdRAG/blob/main/ad_classifier/v5_train_classifier_curriculum_mixed_synthetic_sampling.py))


### 3-2. Fair RAG ([Public Repo](https://github.com/kimdanny/Fair-RAG))
- Rigorous experimental workflow with Stochastic Retriever implementation ([code](https://github.com/kimdanny/Fair-RAG/blob/main/experiment.py))