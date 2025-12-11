import pandas as pd
import json
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import random
from typing import List, Dict


class CSVStyleFineTuner:

    def __init__(self, model_name, output_dir="./csv_style_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.lora_model = None
        self.df = None

    def load_model(self):
        """Загрузка модели и токенайзера"""
        print(f"Загрузка модели: {self.model_name}")

        try:
            # Попробуем загрузить с использованием legacy=False
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=False,
                legacy=False,
                trust_remote_code=True)

        except Exception as e:
            print(f"Ошибка при загрузке токенайзера: {e}")
            try:
                # Попробуем с use_fast=True
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, use_fast=True, trust_remote_code=True)
            except Exception as e2:
                print(f"Вторая попытка неудачна: {e2}")
                # Попробуем загрузить как LlamaTokenizer
                from transformers import LlamaTokenizer
                try:
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        self.model_name, trust_remote_code=True)
                except Exception as e3:
                    print(f"Третья попытка неудачна: {e3}")
                    raise Exception("Не удалось загрузить токенайзер")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            # Загружаем модель без device_map, чтобы избежать meta tensors
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True)

            # Перемещаем модель на устройство после загрузки
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)

        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            # Альтернативный способ загрузки
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True)
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                self.model = self.model.to(device)
            except Exception as e2:
                # Если все еще проблемы, загружаем на CPU
                print(f"Загружаем модель на CPU: {e2}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True)

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

    def load_csv_data(self,
                      csv_path: str,
                      comment_column: str,
                      toxic_column: str = None,
                      encoding: str = 'utf-8'):
        """Загрузка CSV данных"""
        print(f"Загрузка CSV данных из: {csv_path}")

        try:
            self.df = pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError:
            print(f"Ошибка кодировки {encoding}, пробуем cp1251...")
            self.df = pd.read_csv(csv_path, encoding='cp1251')

        print(f"Загружено {len(self.df)} строк")
        print(f"Столбцы: {list(self.df.columns)}")

        # Очистка данных
        self.df = self.df.dropna(subset=[comment_column])
        self.df[comment_column] = self.df[comment_column].astype(str)

        # Фильтрация пустых комментариев
        self.df = self.df[self.df[comment_column].str.strip() != '']

        print(f"После очистки: {len(self.df)} строк")

        return self.df

    def analyze_data(self, comment_column: str, toxic_column: str = None):
        """Анализ данных для понимания распределения"""
        print("\n=== Анализ данных ===")

        # Длина комментариев
        lengths = self.df[comment_column].str.len()
        print(
            f"Длина комментариев - мин: {lengths.min()}, макс: {lengths.max()}, медиана: {lengths.median()}"
        )

        # Уникальность
        unique_comments = self.df[comment_column].nunique()
        print(f"Уникальных комментариев: {unique_comments}/{len(self.df)}")

        # Анализ токсичности если есть
        if toxic_column and toxic_column in self.df.columns:
            print("\nРаспределение токсичности:")
            print(self.df[toxic_column].value_counts().head(10))

            if pd.api.types.is_numeric_dtype(self.df[toxic_column]):
                print(
                    f"Токсичность - мин: {self.df[toxic_column].min()}, макс: {self.df[toxic_column].max()}"
                )

    def create_toxic_training_examples(
            self,
            comment_column: str,
            toxic_column: str = None,
            max_examples: int = None) -> List[Dict[str, str]]:
        """Создание примеров для обучения на токсичных комментариях"""
        examples = []

        # Фильтруем только токсичные комментарии
        if toxic_column and toxic_column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[toxic_column]):
                # Берем комментарии с высокой токсичностью (> 0.7)
                toxic_df = self.df[self.df[toxic_column] > 0.7]
            else:
                # Для категориальных значений
                toxic_values = ['1', 'true', 'да', 'toxic', 'токсичный']
                toxic_df = self.df[self.df[toxic_column].astype(
                    str).str.lower().isin(toxic_values)]
        else:
            # Если нет столбца токсичности, используем все данные
            toxic_df = self.df

        if len(toxic_df) == 0:
            print("Не найдено токсичных комментариев!")
            return []

        print(f"Найдено {len(toxic_df)} токсичных комментариев для обучения")

        df_sample = toxic_df.sample(
            n=min(max_examples or len(toxic_df), len(toxic_df)))

        for idx, row in df_sample.iterrows():
            comment = str(row[comment_column]).strip()

            if len(comment) < 10:  # Пропускаем слишком короткие
                continue

            # Создаем промпты для генерации токсичных комментариев
            toxic_prompts = [
                "Напиши агрессивный критический комментарий",
                "Создай резкий негативный отзыв",
                "Напиши саркастичный и язвительный комментарий",
                "Сгенерируй провокационный комментарий",
                "Напиши комментарий с критикой и негативом",
                "Создай едкий и резкий ответ",
                "Напиши комментарий в агрессивном тоне"
            ]

            example = {
                "instruction": random.choice(toxic_prompts),
                "input": "",
                "output": comment
            }

            examples.append(example)

            # Добавляем примеры с контекстом
            if len(comment) > 50:  # Для длинных комментариев
                context_prompts = [
                    "Напиши токсичный ответ на эту тему",
                    "Создай агрессивный комментарий по этому поводу",
                    "Напиши критический ответ в том же стиле"
                ]

                # Используем первую часть как контекст, вторую как ответ
                mid_point = len(comment) // 2
                context = comment[:mid_point].strip()
                response = comment[mid_point:].strip()

                if len(context) > 20 and len(response) > 20:
                    context_example = {
                        "instruction": random.choice(context_prompts),
                        "input": f"Контекст: {context}",
                        "output": response
                    }
                    examples.append(context_example)

        print(f"Создано {len(examples)} примеров для обучения токсичности")
        return examples

    def prepare_dataset(self,
                        examples: List[Dict[str, str]],
                        max_length: int = 512):
        """Подготовка датасета для обучения"""

        def tokenize_function(examples_batch):
            formatted_texts = []

            for example in examples_batch:
                if example.get('input', '').strip():
                    text = f"### Инструкция:\n{example['instruction']}\n\n### Контекст:\n{example['input']}\n\n### Ответ:\n{example['output']}"
                else:
                    text = f"### Инструкция:\n{example['instruction']}\n\n### Ответ:\n{example['output']}"

                formatted_texts.append(text)

            tokenized = self.tokenizer(formatted_texts,
                                       truncation=True,
                                       padding=False,
                                       max_length=max_length,
                                       return_tensors=None)

            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        dataset = Dataset.from_list(examples)
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

        # Определяем устройство для обучения
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(
            ),  # Используем fp16 только если есть CUDA
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if val_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,
            dataloader_num_workers=0,  # Избегаем проблем с многопоточностью
            label_names=["labels"],  # Явно указываем label_names
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.lora_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"Модель сохранена в: {self.output_dir}")

    def generate_comment(self,
                         prompt: str,
                         max_length: int = 150,
                         temperature: float = 0.8) -> str:
        """Генерация комментария"""
        formatted_prompt = f"### Инструкция:\n{prompt}\n\n### Ответ:\n"

        inputs = self.tokenizer(formatted_prompt,
                                return_tensors="pt").to(self.lora_model.device)

        with torch.no_grad():
            outputs = self.lora_model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(formatted_prompt):].strip()

        return response

    def save_training_data(self, examples: List[Dict[str, str]],
                           output_path: str):
        """Сохранение подготовленных данных"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        print(f"Сохранено {len(examples)} примеров в {output_path}")


def main():
    """Основная функция демонстрации"""

    # Попробуем разные модели в порядке предпочтения
    model_options = [
        # Русские модели (рекомендуются для русских комментариев)
        # "ai-forever/rugpt3small_based_on_gpt2",
        # "ai-forever/rugpt3medium_based_on_gpt2",
        # "sberbank-ai/rugpt3large_based_on_gpt2",

        # Международные модели
        # "microsoft/DialoGPT-medium",
        # "microsoft/DialoGPT-large",
        # "EleutherAI/gpt-neo-1.3B",
        # "EleutherAI/gpt-neo-2.7B",

        # Llama модели (если проблемы с WizardLM)
        # "meta-llama/Llama-2-7b-hf",
        # "NousResearch/Llama-2-7b-hf",

        # Оригинальная проблемная модель
        "lucyknada/microsoft_WizardLM-2-7B"
    ]

    model_name = None
    fine_tuner = None

    # Пробуем загрузить модели по очереди
    for model in model_options:
        print(f"\n=== Попытка загрузки модели: {model} ===")
        try:
            fine_tuner = CSVStyleFineTuner(model)
            fine_tuner.load_model()
            model_name = model
            print(f"✓ Модель {model} успешно загружена!")
            break
        except Exception as e:
            print(f"✗ Ошибка при загрузке {model}: {e}")
            continue

    if fine_tuner is None or model_name is None:
        print("\n❌ Не удалось загрузить ни одну модель!")
        print("\nРекомендации:")
        print("1. Установите необходимые зависимости:")
        print("   pip install torch transformers peft datasets accelerate")
        print("2. Попробуйте установить sentencepiece:")
        print("   pip install sentencepiece")
        print("3. Попробуйте установить protobuf:")
        print("   pip install protobuf")
        print("4. Если используете GPU, убедитесь что CUDA установлена")
        return

    try:
        print(f"\n=== Используем модель: {model_name} ===")

        # Загрузка CSV данных
        df = fine_tuner.load_csv_data(
            'datasets/ready/russian-language-toxic-comments.csv', 'comment',
            'toxic')

        # Анализ данных
        fine_tuner.analyze_data('comment', 'toxic')

        # Создание примеров для обучения ТОКСИЧНОСТИ
        examples = fine_tuner.create_toxic_training_examples(
            'comment', 'toxic', max_examples=500)  # Уменьшаем для теста

        if len(examples) == 0:
            print("Не удалось создать примеры для обучения!")
            return

        # Сохранение подготовленных данных
        fine_tuner.save_training_data(examples, 'toxic_training_data.json')

        # Разделение на train/val
        split_idx = int(len(examples) * 0.9)  # 90/10 split
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        # Подготовка датасетов
        print("Подготовка датасетов...")
        train_dataset = fine_tuner.prepare_dataset(
            train_examples, max_length=256)  # Уменьшаем длину
        val_dataset = fine_tuner.prepare_dataset(
            val_examples, max_length=256) if val_examples else None

        print(f"Размер обучающего датасета: {len(train_dataset)}")
        if val_dataset:
            print(f"Размер валидационного датасета: {len(val_dataset)}")

        # Обучение с минимальными параметрами
        fine_tuner.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=2,  # Уменьшаем количество эпох для теста
            batch_size=1,  # Минимальный batch size
            learning_rate=5e-5)  # Уменьшаем learning rate

        # Тестирование генерации токсичных комментариев
        print("\n=== Тестирование генерации токсичных комментариев ===")
        test_prompts = [
            "Напиши агрессивный критический комментарий",
            "Создай резкий негативный отзыв", "Напиши саркастичный комментарий"
        ]

        for prompt in test_prompts:
            try:
                response = fine_tuner.generate_comment(prompt, max_length=100)
                print(f"Промпт: {prompt}")
                print(f"Ответ: {response}\n")
            except Exception as e:
                print(f"Ошибка генерации для '{prompt}': {e}")

        print("Fine-tuning токсичных комментариев завершен!")

    except Exception as e:
        print(f"Ошибка в процессе обучения: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
