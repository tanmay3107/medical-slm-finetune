# Medical Reasoner: Fine-Tuning Llama 3 for Healthcare 🏥

## Overview
This project fine-tunes a Small Language Model (SLM) on medical Q&A datasets to improve its diagnostic reasoning capabilities without relying on external APIs.

## Tech Stack
* **Model:** Llama 3 8B / Mistral 7B
* **Technique:** QLoRA (Quantized Low-Rank Adaptation)
* **Library:** Unsloth / HuggingFace PEFT
* **Infrastructure:** Google Colab T4 GPU

## Goal
To create a privacy-first, local-runnable model that outperforms the base model on medical queries.