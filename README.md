# TuneWizard AI 🧙‍♂️✨

![image](https://github.com/user-attachments/assets/511f3715-ea60-461e-846e-18f980a3f455)


## Interface

![image](https://github.com/user-attachments/assets/77f36d6c-87fd-425f-b509-c05ef5f4b788)
![image](https://github.com/user-attachments/assets/cac82204-57eb-4b7d-981e-33ec9a4452fc)
![image](https://github.com/user-attachments/assets/279ec58f-f0db-430e-8f8e-b4433b858a58)
![image](https://github.com/user-attachments/assets/83cf3670-5231-49f0-b6e0-b32e8a09b098)
![image](https://github.com/user-attachments/assets/515fd359-73a4-443b-94e7-b57a57a6d8cd)
![image](https://github.com/user-attachments/assets/9edc8d15-1404-4057-9b8e-0f2a60a35e29)



TuneWizard AI is an intuitive, GUI-based tool for fine-tuning Large Language Models (LLMs). It makes the complex process of LLM fine-tuning accessible to everyone, from AI enthusiasts to professional researchers.

## 🌟 Features

- 🖥️ User-friendly GUI interface
- 📊 Support for various open-source LLM models
- 📁 Flexible dataset input (Hugging Face links or CSV uploads)
- 🎛️ Interactive hyperparameter tuning
- 📈 Real-time training visualizations
- 🧪 Comprehensive model evaluation metrics
- 🚀 Easy model export and deployment

🚀 Quick Start
Prerequisites

Google Colab account or Kaggle account
GPU runtime (T4 GPU on Colab or GPU on Kaggle)

Running the Notebook

Open the TuneWizard AI notebook in Google Colab or Kaggle.
Ensure you have selected a GPU runtime:

For Google Colab: Runtime > Change runtime type > Hardware accelerator > GPU
For Kaggle: Settings > Accelerator > GPU


Run the notebook cells in order, following the instructions provided in each cell.
When prompted, upload your dataset or provide a Hugging Face dataset link.
Adjust hyperparameters as needed using the provided interface.
Start the training process and monitor the results in real-time.

Note: The notebook will install all necessary dependencies automatically. No manual installation is required.
## 💡 How to Use

1. **Select Dataset**: Choose between entering a Hugging Face dataset link or uploading a CSV file(Coming soon). Make sure your dataset columns should have these Headers: Question, Title and Answers, Adjust in case of conflicts in  .

2. **Choose Model**: Select from our list of supported models or enter a custom model name (Must be supported by unsloth please check here: https://github.com/unslothai/unsloth ).

3. **Set Hyperparameters**: Adjust batch size, learning rate, warmup steps, and more using intuitive sliders.

4. **Train Model**: Click "Train and Evaluate" to start the fine-tuning process.

5. **Evaluate Results**: Review performance metrics and visualizations to assess your model's quality.

6. **Run Inference**: Test your fine-tuned model with custom inputs in the Inference tab.

## 📊 Supported Models

- Llama 3.1 (8B)
- Mistral Nemo (12B)
- Gemma 2 (9B)
- Phi-3 (mini)
- Ollama
- Mistral v0.3 (7B)
- QRPO
- DPO Zephyr
- TinyLlama
- ... and more!

## 🛠️ Advanced Configuration

For advanced users, TuneWizard AI supports custom configurations through a `config.yaml` file. See our [Advanced Configuration Guide](docs/advanced-config.md) for more details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details on how to get started.

## 📜 License

TuneWizard AI is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co/) for their amazing transformers library
- [Gradio](https://gradio.app/) for making GUI creation a breeze
- All the open-source LLM creators and contributors

## 📞 Contact

For support or queries, please open an issue or contact us at support@tunewizardai.com.

---

Made with ❤️ by [Jaskirat](https://github.com/Jaskirat-singh04)
