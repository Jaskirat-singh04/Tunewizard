 import os
import gradio as gr
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import csv
import random
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

# Utility functions
def preprocess_text(text):
    return word_tokenize(text.lower())

def calculate_bleu(reference, candidate):
    return sentence_bleu([preprocess_text(reference)], preprocess_text(candidate))

def calculate_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)[0]
    return {
        'rouge-1': scores['rouge-1']['f'],
        'rouge-2': scores['rouge-2']['f'],
        'rouge-l': scores['rouge-l']['f'],
    }

def calculate_rouge_w(reference, candidate, weight=1.2):
    def lcs(X, Y):
        m, n = len(X), len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i-1] == Y[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        return L[m][n]

    ref_tokens = preprocess_text(reference)
    cand_tokens = preprocess_text(candidate)

    lcs_length = lcs(ref_tokens, cand_tokens)
    weighted_lcs = lcs_length ** weight

    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return 0.0

    r_lcs = weighted_lcs / (len(ref_tokens) ** weight)
    p_lcs = weighted_lcs / (len(cand_tokens) ** weight)

    if r_lcs == 0 or p_lcs == 0:
        return 0.0

    beta = p_lcs / r_lcs
    f_lcs = ((1 + beta**2) * r_lcs * p_lcs) / (r_lcs + beta**2 * p_lcs)

    return f_lcs

def exact_match(reference, candidate):
    return int(reference.strip().lower() == candidate.strip().lower())

# Model training function
def train_model(dataset_link, model_name, batch_size, grad_accum_steps, warmup_steps, max_steps, learning_rate, weight_decay):
    # Load dataset
    dataset = load_dataset(dataset_link, split="train")

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=weight_decay,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    )

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"unsloth/{model_name}",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Format dataset
    def formatting_prompts_func(examples):
        instructions = examples["Question"]
        inputs = examples["Title"]
        outputs = examples["Answer"]
        texts = [
            f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}{tokenizer.eos_token}"""
            for instruction, input, output in zip(instructions, inputs, outputs)
        ]
        return {"text": texts}

    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    train_dataset = formatted_dataset.select(range(int(0.9 * len(formatted_dataset))))
    eval_dataset = formatted_dataset.select(range(int(0.9 * len(formatted_dataset)), len(formatted_dataset)))

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        args=training_args,
    )

    # Train the model
    trainer.train()

    return model, tokenizer

def inference(model, tokenizer, instruction, input_text):
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Disable use_cache and add error handling
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=False,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return response.split("### Response:")[-1].strip()
    except Exception as e:
        return f"An error occurred during inference: {str(e)}"

# Model evaluation function
def evaluate_model(model, tokenizer, dataset, num_samples):
    samples = random.sample(list(zip(dataset['Question'], dataset['Title'], dataset['Answer'])), num_samples)

    results = []
    for question, title, answer in samples:
        response = inference(model, tokenizer, question, title)

        bleu = calculate_bleu(answer, response)
        rouge_scores = calculate_rouge(answer, response)
        rouge_w = calculate_rouge_w(answer, response)
        exact_match_score = exact_match(answer, response)

        results.append({
            'Question': question,
            'Answer': answer,
            'Response': response,
            'BLEU': bleu,
            'ROUGE-1': rouge_scores['rouge-1'],
            'ROUGE-2': rouge_scores['rouge-2'],
            'ROUGE-L': rouge_scores['rouge-l'],
            'ROUGE-W': rouge_w,
            'Exact Match': exact_match_score
        })

    return pd.DataFrame(results)

# Visualization function
def create_visualizations(df):
    metrics = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-W', 'Exact Match']

    # Bar Plot for Average Scores
    avg_scores = [df[metric].mean() for metric in metrics]
    fig_bar = px.bar(x=metrics, y=avg_scores, title='Average Evaluation Scores',
                     labels={'x': 'Metrics', 'y': 'Average Score'},
                     color=metrics, color_continuous_scale='Viridis')
    fig_bar.update_layout(xaxis_title='Metrics', yaxis_title='Score', yaxis=dict(range=[0, 1]), showlegend=False)

    # Distribution Plots
    fig_dist = px.histogram(df, x=metrics, marginal="box", title='Distribution of Scores for Each Metric', barmode='overlay')
    fig_dist.update_layout(bargap=0.2)

    # Box Plots
    fig_box = go.Figure()
    for metric in metrics:
        fig_box.add_trace(go.Box(y=df[metric], name=metric))
    fig_box.update_layout(title='Distribution of Evaluation Metrics', yaxis_title='Score', xaxis_title='Metrics')

    # Scatter Matrix
    fig_scatter = px.scatter_matrix(df, dimensions=metrics, title='Pairwise Scatter Matrix of Evaluation Metrics')

    # Line Graph
    fig_line = go.Figure()
    for metric in metrics:
        fig_line.add_trace(go.Scatter(x=df.index, y=df[metric], mode='lines', name=metric))
    fig_line.update_layout(title='Evaluation Metrics Over Instances', xaxis_title='Instance Index', yaxis_title='Score')

    return fig_bar, fig_dist, fig_box, fig_scatter, fig_line

def train_and_evaluate(dataset_link, model_name, batch_size, grad_accum_steps, warmup_steps, max_steps, learning_rate, weight_decay, num_eval_samples):
    try:
        model, tokenizer = train_model(dataset_link, model_name, batch_size, grad_accum_steps, warmup_steps, max_steps, learning_rate, weight_decay)

        dataset = load_dataset(dataset_link, split="test")
        eval_results = evaluate_model(model, tokenizer, dataset, num_eval_samples)

        mean_scores = eval_results[['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-W', 'Exact Match']].mean().to_frame().T

        visualizations = create_visualizations(eval_results)

        return model, tokenizer, eval_results, mean_scores, visualizations
    except Exception as e:
        error_message = f"An error occurred during training and evaluation: {str(e)}"
        return None, None, pd.DataFrame(), pd.DataFrame(), (None, None, None, None, None)


# Gradio interface
def gradio_interface():
    model_state = {"model": None, "tokenizer": None}

    def train_and_update(dataset_link, model_name, batch_size, grad_accum_steps, warmup_steps, max_steps, learning_rate, weight_decay, num_eval_samples):
        model, tokenizer, eval_results, mean_scores, visualizations = train_and_evaluate(
        dataset_link, model_name, batch_size, grad_accum_steps, warmup_steps, max_steps, learning_rate, weight_decay, num_eval_samples
    )
        if model is not None and tokenizer is not None:
          model_state["model"] = model
          model_state["tokenizer"] = tokenizer
        return eval_results, mean_scores, *visualizations

    def run_inference(instruction, input_text):
        if model_state["model"] is None or model_state["tokenizer"] is None:
            return "Please train the model first."
        return inference(model_state["model"], model_state["tokenizer"], instruction, input_text)

    with gr.Blocks() as demo:
        gr.Markdown("# LLM Fine-tuning and Evaluation Interface")
        gr.Markdown("Welcome to the LLM Fine-tuning and Evaluation Interface. Follow the steps below to train, evaluate, and interact with your model.")


        with gr.Tab("Instructions & Guidelines"):
            gr.Markdown("## Welcome to the LLM Fine-tuning and Evaluation App")

            gr.Markdown(
                """
                ### How to Use This App

                1. **Dataset Selection**
                   - In the "Training and Evaluation" tab, enter the Hugging Face dataset link or use the default one provided.
                   - The default dataset is "FinLang/investopedia-embedding-dataset".

                2. **Model Selection**
                   - Choose the model you want to fine-tune from the dropdown menu.
                   - Currently, "llama-3-8b-bnb-4bit" is available.

                3. **Hyperparameter Setting**
                   - Adjust the training hyperparameters using the sliders and input boxes:
                     - Batch Size: Number of samples processed before the model is updated.
                     - Gradient Accumulation Steps: Number of steps to accumulate gradients before performing a backward/update pass.
                     - Warmup Steps: Number of steps for the learning rate warmup.
                     - Max Steps: Total number of training steps.
                     - Learning Rate: Step size at each iteration while moving toward a minimum of a loss function.
                     - Weight Decay: L2 regularization term.

                4. **Evaluation Setup**
                   - Set the number of evaluation samples to use after training.

                5. **Training and Evaluation**
                   - Click the "Train and Evaluate" button to start the process.
                   - Wait for the training to complete and evaluation results to appear.

                6. **Reviewing Results**
                   - Examine the evaluation results in the provided tables and plots.
                   - You can download the detailed evaluation results using the "Download Evaluation Results" button.

                7. **Inference**
                   - Switch to the "Inference" tab to test your fine-tuned model.
                   - Enter an instruction and input text, then click "Generate Response" to see the model's output.

                ### Tips
                - Fine-tuning can take a while, especially for larger datasets or more training steps.
                - Experiment with different hyperparameters to optimize model performance.
                - Always evaluate your model's performance before deploying it in any application.

                ### Troubleshooting
                - If you encounter any errors during training or inference, check the error message for details.
                - Ensure you have a stable internet connection, especially when downloading models and datasets.
                - If issues persist, try refreshing the page or restarting the runtime.

                Enjoy using the LLM Fine-tuning and Evaluation App!
                """
            )
        with gr.Tab("Training and Evaluation"):
            dataset_link = gr.Textbox(label="Dataset Link (Hugging Face)", value="FinLang/investopedia-embedding-dataset")
            model_name = gr.Dropdown(["llama-3-8b-bnb-4bit"], label="Model Name", value="llama-3-8b-bnb-4bit")

            with gr.Row():
                batch_size = gr.Slider(1, 8, value=2, step=1, label="Batch Size")
                grad_accum_steps = gr.Slider(1, 16, value=8, step=1, label="Gradient Accumulation Steps")

            with gr.Row():
                warmup_steps = gr.Slider(0, 100, value=20, step=1, label="Warmup Steps")
                max_steps = gr.Slider(0, 500, value=5, step=10, label="Max Steps")

            with gr.Row():
                learning_rate = gr.Number(value=5e-5, label="Learning Rate")
                weight_decay = gr.Number(value=0.01, label="Weight Decay")

            num_eval_samples = gr.Slider(5, 50, value=10, step=1, label="Number of Evaluation Samples")

            train_button = gr.Button("Train and Evaluate")

            eval_results = gr.Dataframe(label="Evaluation Results")
            mean_scores = gr.Dataframe(label="Mean Scores")

            with gr.Row():
                fig_bar = gr.Plot(label="Average Evaluation Scores")
                fig_dist = gr.Plot(label="Distribution of Scores")

            with gr.Row():
                fig_box = gr.Plot(label="Box Plots of Evaluation Metrics")
                fig_scatter = gr.Plot(label="Scatter Matrix of Evaluation Metrics")

            fig_line = gr.Plot(label="Evaluation Metrics Over Instances")

            download_button = gr.Button("Download Evaluation Results")

        with gr.Tab("Inference"):
            instruction = gr.Textbox(label="Instruction")
            input_text = gr.Textbox(label="Input")
            inference_button = gr.Button("Generate Response")
            response = gr.Textbox(label="Model Response")

        # Event handlers
        train_button.click(
            train_and_update,
            inputs=[dataset_link, model_name, batch_size, grad_accum_steps, warmup_steps, max_steps, learning_rate, weight_decay, num_eval_samples],
            outputs=[eval_results, mean_scores, fig_bar, fig_dist, fig_box, fig_scatter, fig_line]
        )

        download_button.click(
            lambda df: df.to_csv("evaluation_results.csv", index=False),
            inputs=[eval_results],
            outputs=None
        )

        inference_button.click(
            run_inference,
            inputs=[instruction, input_text],
            outputs=[response]
        )

    return demo

# Launch the Gradio interface
demo = gradio_interface()
demo.launch(share=True, debug=True)