import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import os


class DialogLoRAFineTuner:

    def __init__(self, model_name, output_dir="./dialog_lora_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.lora_model = None

        # Специальные токены для диалогов
        self.special_tokens = {
            "system": "<|system|>",
            "user": "<|user|>",
            "assistant": "<|assistant|>",
            "end": "<|end|>"
        }

    def load_model(self):
        """Загрузка модели и токенайзера с поддержкой диалогов"""
        print(f"Загрузка модели: {self.model_name}")

        # Загрузка токенайзера
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Добавляем специальные токены
        special_tokens_dict = {
            "additional_special_tokens": list(self.special_tokens.values())
        }

        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = "<|pad|>"

        num_added_tokens = self.tokenizer.add_special_tokens(
            special_tokens_dict)

        # Загрузка модели
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True)

        # Расширяем embedding слой если добавили токены
        if num_added_tokens > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Настройка LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj",
                "down_proj"
            ],
            bias="none",
        )

        self.lora_model = get_peft_model(self.model, lora_config)

        print(f"Trainable parameters: {self.lora_model.num_parameters()}")

    def prepare_dialog_dataset(self,
                               data_path,
                               max_length=1024,
                               data_format="instruction"):
        """Подготовка диалогового датасета"""
        print(f"Загрузка диалоговых данных из: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if data_format == "instruction":
            return self._prepare_instruction_dataset(data, max_length)
        elif data_format == "conversation":
            return self._prepare_conversation_dataset(data, max_length)
        else:
            raise ValueError(
                "data_format должен быть 'instruction' или 'conversation'")

    def _prepare_instruction_dataset(self, data, max_length):
        """Подготовка датасета в формате instruction"""

        def tokenize_function(examples):
            # Форматируем каждый пример
            formatted_texts = []
            for example in examples:
                # Создаем полный текст для обучения
                input_text = example.get('input', '')
                if input_text:
                    full_text = f"{example['instruction']}\n\n{input_text}\n\n{example['output']}"
                else:
                    full_text = f"{example['instruction']}\n\n{example['output']}"

                formatted_texts.append(full_text)

            # Токенизация
            tokenized = self.tokenizer(formatted_texts,
                                       truncation=True,
                                       padding=False,
                                       max_length=max_length,
                                       return_tensors=None)

            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        dataset = Dataset.from_list(data)
        return dataset.map(lambda x: tokenize_function([x]),
                           batched=False,
                           remove_columns=dataset.column_names)

    def _prepare_conversation_dataset(self, data, max_length):
        """Подготовка датасета в формате conversation"""

        def format_conversation(example):
            """Форматирование разговора в одну строку"""
            formatted = ""
            for message in example["messages"]:
                role = message["role"]
                content = message["content"]

                if role == "system":
                    formatted += f"{self.special_tokens['system']}\n{content}{self.special_tokens['end']}\n"
                elif role == "user":
                    formatted += f"{self.special_tokens['user']}\n{content}{self.special_tokens['end']}\n"
                elif role == "assistant":
                    formatted += f"{self.special_tokens['assistant']}\n{content}{self.special_tokens['end']}\n"

            return formatted

        def tokenize_function(examples):
            formatted_texts = [format_conversation(ex) for ex in examples]

            tokenized = self.tokenizer(formatted_texts,
                                       truncation=True,
                                       padding=False,
                                       max_length=max_length,
                                       return_tensors=None)

            # Для диалогов мы хотим обучать только на ответах ассистента
            # Можно реализовать маскирование, но для простоты пока обучаем на всем
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        dataset = Dataset.from_list(data)
        return dataset.map(lambda x: tokenize_function([x]),
                           batched=False,
                           remove_columns=dataset.column_names)

    def train(self,
              train_dataset,
              val_dataset=None,
              epochs=3,
              batch_size=2,
              learning_rate=2e-4):
        """Обучение диалоговой модели"""
        print("Начало обучения диалоговой модели...")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,
            warmup_steps=100,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if val_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Отключаем wandb
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, pad_to_multiple_of=8)

        trainer = Trainer(
            model=self.lora_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"Модель сохранена в: {self.output_dir}")

    def test_generation(self, prompt, max_length=512):
        """Тестирование генерации после обучения"""
        inputs = self.tokenizer(prompt,
                                return_tensors="pt").to(self.lora_model.device)

        with torch.no_grad():
            outputs = self.lora_model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()


def main():
    """Демонстрация обучения на диалоговых данных"""

    # Настройки
    model_name = "microsoft/DialoGPT-medium"  # Или другая модель

    # Инициализация
    fine_tuner = DialogLoRAFineTuner(model_name)

    # Загрузка модели
    fine_tuner.load_model()

    # Подготовка данных (используем instruction формат)
    train_dataset = fine_tuner.prepare_dialog_dataset(
        "dialog_instruction_chatml.json", data_format="instruction")

    # Обучение
    fine_tuner.train(
        train_dataset=train_dataset,
        epochs=3,
        batch_size=2,  # Уменьшаем из-за длинных диалогов
        learning_rate=2e-4)

    # Тестирование
    print("\n=== Тестирование модели ===")
    test_prompt = f"{fine_tuner.special_tokens['user']}\nПривет! Как дела?{fine_tuner.special_tokens['end']}\n{fine_tuner.special_tokens['assistant']}\n"

    response = fine_tuner.test_generation(test_prompt)
    print(f"Промпт: {test_prompt}")
    print(f"Ответ: {response}")

    print("Fine-tuning завершен!")


if __name__ == "__main__":
    main()
