import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json


class LoRAFineTuner:

    def __init__(self, model_name, output_dir="./lora_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.lora_model = None

    def load_model(self):
        """Загрузка модели и токенайзера"""
        print(f"Загрузка модели: {self.model_name}")

        # Загрузка токенайзера
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Добавляем pad_token если его нет
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Загрузка модели
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True)

        # Настройка LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # rank
            lora_alpha=32,  # alpha
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj",
                "down_proj"
            ],
            bias="none",
        )

        self.lora_model = get_peft_model(self.model, lora_config)

        print(f"Trainable parameters: {self.lora_model.num_parameters()}")
        print(
            f"All parameters: {self.lora_model.num_parameters(only_trainable=False)}"
        )

    def prepare_dataset(self, data_path, max_length=512):
        """Подготовка датасета"""
        print(f"Загрузка данных из: {data_path}")

        # Загрузка данных
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)

        # Форматирование данных
        def format_instruction(example):
            if example.get('input', '').strip():
                return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
            else:
                return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

        # Токенизация
        def tokenize_function(examples):
            formatted_texts = [format_instruction(ex) for ex in examples]

            # Токенизация
            tokenized = self.tokenizer(formatted_texts,
                                       truncation=True,
                                       padding=False,
                                       max_length=max_length,
                                       return_tensors=None)

            # Установка labels для обучения
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        # Создание датасета
        dataset = Dataset.from_list(data)
        tokenized_dataset = dataset.map(lambda x: tokenize_function([x]),
                                        batched=False,
                                        remove_columns=dataset.column_names)

        return tokenized_dataset

    def train(self,
              train_dataset,
              val_dataset=None,
              epochs=3,
              batch_size=4,
              learning_rate=2e-4):
        """Обучение модели"""
        print("Начало обучения...")

        # Настройки обучения
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if val_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, pad_to_multiple_of=8)

        # Trainer
        trainer = Trainer(
            model=self.lora_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Обучение
        trainer.train()

        # Сохранение модели
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"Модель сохранена в: {self.output_dir}")

    def merge_and_save(self, output_path="./merged_model"):
        """Объединение LoRA с базовой моделью"""
        print("Объединение LoRA адаптера с базовой моделью...")

        # Загрузка обученной модели
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto")

        lora_model = PeftModel.from_pretrained(base_model, self.output_dir)
        merged_model = lora_model.merge_and_unload()

        # Сохранение объединенной модели
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        print(f"Объединенная модель сохранена в: {output_path}")


def main():
    # Пример использования

    # 1. Инициализация
    model_name = "lmstudio-community/WizardLM-2-7B-GGUF"  # Замените на вашу модель
    fine_tuner = LoRAFineTuner(model_name)

    # 2. Загрузка модели
    fine_tuner.load_model()

    # 3. Подготовка данных
    train_dataset = fine_tuner.prepare_dataset("train_data.json")
    val_dataset = fine_tuner.prepare_dataset("val_data.json")  # Опционально

    # 4. Обучение
    fine_tuner.train(train_dataset=train_dataset,
                     val_dataset=val_dataset,
                     epochs=3,
                     batch_size=2,
                     learning_rate=2e-4)

    # 5. Объединение и сохранение (опционально)
    fine_tuner.merge_and_save("./final_model")

    print("Fine-tuning завершен!")


# Пример создания данных для обучения
# def create_sample_data():
#     """Создание примера данных для обучения"""
#     sample_data = [{
#         "instruction": "Переведи на русский язык",
#         "input": "Hello, how are you?",
#         "output": "Привет, как дела?"
#     }, {
#         "instruction":
#         "Объясни что такое машинное обучение",
#         "input":
#         "",
#         "output":
#         "Машинное обучение - это область искусственного интеллекта, которая позволяет компьютерам учиться и принимать решения на основе данных без явного программирования."
#     }, {
#         "instruction": "Реши математическую задачу",
#         "input": "2 + 2 = ?",
#         "output": "2 + 2 = 4"
#     }]

#     with open("train_data.json", "w", encoding="utf-8") as f:
#         json.dump(sample_data, f, ensure_ascii=False, indent=2)

#     print("Пример данных создан: train_data.json")

if __name__ == "__main__":
    # Создание примера данных
    # create_sample_data()

    # Запуск обучения
    main()
