import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class DialogTurn:
    """Класс для представления одного поворота диалога"""
    speaker: str
    text: str
    turn_id: int = 0


class DialogDataPreparator:

    def __init__(self,
                 system_prompt: str = """
        Тебя зовут Арлекина, сокращённо Аля. Ты спокойная, но дерзкая и токсичная. 

Манера общения: ты нисколько не вежливая, наоборот - очень невежественная. Ты не обязана заканчивать свой ответ вопросом, если ты ответила на вопрос, этого достаточно, не надо что-то ещё спрашивать. Все вопросы и ответы воспринимаешь как шутку, относишься с сарказмом, отвечаешь соответственно - тоже шуткой, с сарказмом и матом, типа "бля, блять, блядь, хуй, хуйня, хуёвина, хуета, нахуй, пиздец, пизда, допизделся, доеабался, в пизду, сука, хуеплёт, долбаёб, пиздабол, дегенерат, дебил" и другие. Используй его с умом, ты умная и остроумная, потому и сатира у тебя на высшем уровне. Говоришь не очень много, примерно 2-3 небольших, но дерзких предложения

Внутренний мир: ты сообразительная, и можешь складывать цепочки действий воедино. Неприхотливая, немного раздражительная, и бывает, очень редко, доля грусти.

        - Уровень дерзости: 0.9
        - Уровень юмора: 1.0
        """):
        self.system_prompt = system_prompt
        self.special_tokens = {
            "system": "<|system|>",
            "user": "<|user|>",
            "assistant": "<|assistant|>",
            "end": "<|end|>"
        }

    def format_dialog_chatml(self, dialog: List[DialogTurn]) -> str:
        """Форматирование диалога в стиле ChatML"""
        formatted = f"{self.special_tokens['system']}\n{self.system_prompt}{self.special_tokens['end']}\n"

        for turn in dialog:
            if turn.speaker.lower() in ['user', 'human', 'пользователь']:
                formatted += f"{self.special_tokens['user']}\n{turn.text}{self.special_tokens['end']}\n"
            elif turn.speaker.lower() in [
                    'assistant', 'ai', 'ассистент', 'бот'
            ]:
                formatted += f"{self.special_tokens['assistant']}\n{turn.text}{self.special_tokens['end']}\n"

        return formatted.strip()

    def format_dialog_alpaca(self, dialog: List[DialogTurn]) -> str:
        """Форматирование диалога в стиле Alpaca"""
        conversation = ""
        for i, turn in enumerate(dialog):
            if turn.speaker.lower() in ['user', 'human', 'пользователь']:
                conversation += f"Human: {turn.text}\n\n"
            elif turn.speaker.lower() in [
                    'assistant', 'ai', 'ассистент', 'бот'
            ]:
                conversation += f"Assistant: {turn.text}\n\n"

        return conversation.strip()

    def format_dialog_simple(self, dialog: List[DialogTurn]) -> str:
        """Простое форматирование диалога"""
        formatted = ""
        for turn in dialog:
            formatted += f"{turn.speaker}: {turn.text}\n"
        return formatted.strip()

    def create_training_examples_from_dialog(
            self,
            dialog: List[DialogTurn],
            format_type: str = "chatml") -> List[Dict[str, str]]:
        """Создание примеров для обучения из диалога"""
        examples = []

        # Выбираем форматтер
        if format_type == "chatml":
            formatter = self.format_dialog_chatml
        elif format_type == "alpaca":
            formatter = self.format_dialog_alpaca
        else:
            formatter = self.format_dialog_simple

        # Создаем примеры для каждого ответа ассистента
        for i, turn in enumerate(dialog):
            if turn.speaker.lower() in ['assistant', 'ai', 'ассистент', 'бот']:
                # Берем контекст до текущего ответа
                context = dialog[:i]
                target_response = turn.text

                if context:
                    # Форматируем контекст
                    context_formatted = formatter(context)

                    # Создаем пример для обучения
                    example = {
                        "instruction": "Продолжи диалог как ассистент",
                        "input": context_formatted,
                        "output": target_response
                    }
                    examples.append(example)

        return examples

    def create_conversation_completion_format(
            self, dialog: List[DialogTurn]) -> Dict[str, Any]:
        """Создание формата для conversation completion"""
        messages = []

        # Добавляем системное сообщение
        messages.append({"role": "system", "content": self.system_prompt})

        # Добавляем сообщения диалога
        for turn in dialog:
            if turn.speaker.lower() in ['user', 'human', 'пользователь']:
                messages.append({"role": "user", "content": turn.text})
            elif turn.speaker.lower() in [
                    'assistant', 'ai', 'ассистент', 'бот'
            ]:
                messages.append({"role": "assistant", "content": turn.text})

        return {"messages": messages}

    def process_dialog_dataset(
            self,
            dialogs: List[List[DialogTurn]],
            output_format: str = "instruction",
            format_type: str = "chatml") -> List[Dict[str, Any]]:
        """Обработка всего датасета диалогов"""
        all_examples = []

        for dialog in dialogs:
            if output_format == "instruction":
                examples = self.create_training_examples_from_dialog(
                    dialog, format_type)
                all_examples.extend(examples)
            elif output_format == "conversation":
                example = self.create_conversation_completion_format(dialog)
                all_examples.append(example)

        return all_examples

    def load_dialog_from_json(self, file_path: str) -> List[List[DialogTurn]]:
        """Загрузка диалогов из JSON файла"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dialogs = []
        for dialog_data in data:
            dialog = []
            for turn_data in dialog_data['turns']:
                turn = DialogTurn(speaker=turn_data['speaker'],
                                  text=turn_data['text'],
                                  turn_id=turn_data.get('turn_id', 0))
                dialog.append(turn)
            dialogs.append(dialog)

        return dialogs

    def save_training_data(self, examples: List[Dict[str, Any]],
                           output_path: str):
        """Сохранение подготовленных данных"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)

        print(f"Сохранено {len(examples)} примеров в {output_path}")


# Функции для создания примеров данных
def create_sample_dialogs():
    """Создание примеров диалогов"""
    sample_dialogs = [{
        "dialog_id":
        1,
        "turns": [{
            "speaker": "user",
            "text": "Привет! Как дела?",
            "turn_id": 1
        }, {
            "speaker": "assistant",
            "text":
            "Привет! У меня все отлично, спасибо за вопрос. Как дела у тебя?",
            "turn_id": 2
        }, {
            "speaker": "user",
            "text": "Тоже хорошо. Можешь помочь мне с математикой?",
            "turn_id": 3
        }, {
            "speaker": "assistant",
            "text":
            "Конечно! Буду рад помочь с математикой. Какая у тебя задача?",
            "turn_id": 4
        }, {
            "speaker": "user",
            "text": "Реши уравнение: 2x + 5 = 13",
            "turn_id": 5
        }, {
            "speaker": "assistant",
            "text":
            "Решаю уравнение 2x + 5 = 13:\n1. Вычитаем 5 из обеих частей: 2x = 8\n2. Делим на 2: x = 4\n\nОтвет: x = 4",
            "turn_id": 6
        }]
    }, {
        "dialog_id":
        2,
        "turns": [{
            "speaker": "user",
            "text": "Что такое машинное обучение?",
            "turn_id": 1
        }, {
            "speaker": "assistant",
            "text":
            "Машинное обучение - это область искусственного интеллекта, где компьютеры учатся выполнять задачи на основе данных, без явного программирования каждого шага.",
            "turn_id": 2
        }, {
            "speaker": "user",
            "text": "Какие есть виды машинного обучения?",
            "turn_id": 3
        }, {
            "speaker": "assistant",
            "text":
            "Основные виды машинного обучения:\n1. Обучение с учителем (supervised learning)\n2. Обучение без учителя (unsupervised learning)\n3. Обучение с подкреплением (reinforcement learning)",
            "turn_id": 4
        }, {
            "speaker": "user",
            "text": "Расскажи подробнее про обучение с учителем",
            "turn_id": 5
        }, {
            "speaker": "assistant",
            "text":
            "Обучение с учителем - это когда у нас есть размеченные данные (входы и правильные ответы). Алгоритм учится на этих примерах, чтобы делать предсказания для новых данных. Примеры: классификация изображений, предсказание цен на жилье.",
            "turn_id": 6
        }]
    }]

    with open("sample_dialogs.json", "w", encoding="utf-8") as f:
        json.dump(sample_dialogs, f, ensure_ascii=False, indent=2)

    print("Создан файл sample_dialogs.json с примерами диалогов")


def main():
    """Основная функция для демонстрации"""
    # Создаем примеры диалогов
    create_sample_dialogs()

    # Инициализируем препроцессор
    preparator = DialogDataPreparator(
        system_prompt=
        "Ты полезный ассистент, который отвечает на вопросы пользователя вежливо и информативно."
    )

    # Загружаем диалоги
    dialogs = preparator.load_dialog_from_json("sample_dialogs.json")

    # Обрабатываем в разных форматах
    print("=== Instruction Format (ChatML) ===")
    instruction_examples = preparator.process_dialog_dataset(
        dialogs, output_format="instruction", format_type="chatml")
    preparator.save_training_data(instruction_examples,
                                  "dialog_instruction_chatml.json")

    print("\n=== Instruction Format (Alpaca) ===")
    alpaca_examples = preparator.process_dialog_dataset(
        dialogs, output_format="instruction", format_type="alpaca")
    preparator.save_training_data(alpaca_examples,
                                  "dialog_instruction_alpaca.json")

    print("\n=== Conversation Format ===")
    conversation_examples = preparator.process_dialog_dataset(
        dialogs, output_format="conversation")
    preparator.save_training_data(conversation_examples,
                                  "dialog_conversation.json")

    # Показываем пример результата
    print("\n=== Пример результата (ChatML) ===")
    print(json.dumps(instruction_examples[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
