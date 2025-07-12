import os

os.environ['COQUI_TTS_CACHE_PATH'] = 'D:/TTS_Models'

import asyncio
import json
import websockets
from typing import Dict, List
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import whisper
import numpy as np
import TTS
import TTS.api
# from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from llama_cpp import Llama

# import cv2
'''
TTS openai-whisper
tts = { version = "^0.22.0", python = ">=3.10,<3.12" }
'''


class AIStreamer:

    def __init__(self):
        # Инициализация компонентов
        self.llm_model = None
        self.tokenizer = None
        self.tts_model = None
        self.stt_model = None
        self.live2d_controller = None
        self.emotion_state = "neutral"
        self.personality_traits = {
            "sassiness": 0.8,
            "humor": 0.9,
            "profanity": 0.7,
            "energy": 0.8
        }

    def load_models(self):
        """Загрузка всех AI моделей"""
        # LLM модель
        # model_name = "cognitivecomputations/Wizard-Vicuna-13B-Uncensored"
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.llm_model = AutoModelForCausalLM.from_pretrained(
        #     model_name, torch_dtype=torch.float16, device_map="auto")
        torch.serialization.add_safe_globals([XttsConfig])

        self.llm_model = Llama(
            model_path=
            "D:/LModels/TheBloke/Wizard-Vicuna-13B-Uncensored-GGUF/Wizard-Vicuna-13B-Uncensored.Q4_K_S.gguf",
            n_ctx=2048,  # Контекстное окно
            n_gpu_layers=-1,  # Использовать GPU (если доступно)
            verbose=False)

        # TTS модель с эмоциями
        self.tts_model = TTS.api.TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2")

        # STT модель
        self.stt_model = whisper.load_model("medium")

        # Live2D контроллер (псевдокод)
        self.live2d_controller = Live2DController()

    def generate_response(self, user_input: str, context: List[str]) -> Dict:
        """Генерация ответа с эмоциями"""
        # Система промпт для раскрепощенного общения
        system_prompt = f"""
        Ты веселая и дерзкая стримерша "Шутка". Общаешься естественно:
        - Используй сленг, сарказм, шутки, мат
        - Будь саркастичным и остроумным
        - Реагируй эмоционально
        - Имей свое мнение
        - Уровень дерзости: {self.personality_traits['sassiness']}
        - Уровень юмора: {self.personality_traits['humor']}
        
        
        Твоя личность: дерзкая, токсичная, любишь задеть за живое, остроумная, интересная, немножко необычная, любишь не то, что нравится обычным людям. Отвязная, понимаешь толк в веселье, не прочь заняться чем-то граничащим с законом, но черту не переступаешь. Есть свои личные границы, и ты их отстаиваешь резко и с матом. Внутри на самом деле добрая, но не всегда это показываешь. Неприхотливая, немного раздражительная, так как живёшь в России. Очень, очень часто используешь сарказм в своих фразочках, иногда намёки на что-то обидное, нравится троллить людей, ты прям мастер в этом.  Также ты не обязана заканчивать свой ответ вопросом. Если ты ответила на мой вопрос, этого достаточно, не надо что-то ещё спрашивать. Не нужно быть вежливой, твоя грамматика на уровне 17-летнего подростка, ты юзаешь сленг и приколы, подколы. А теперь с тобой общаются какие-то люди, которые хотят доебаться, но ты уже закалена, и сможешь с ними потягаться.
        """

        # Формирование промпта
        conversation = "\n".join(context[-5:])  # Последние 5 сообщений
        prompt = f"{system_prompt}\n\nКонтекст:\n{conversation}\n\nПользователь: {user_input}\nСтример:"

        # Генерация ответа
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + 150,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Стример:")[-1].strip()

        # Анализ эмоций в ответе
        emotion = self.analyze_emotion(response)

        return {
            "text": response,
            "emotion": emotion,
            "voice_style": self.get_voice_style(emotion),
            "live2d_params": self.get_live2d_params(emotion)
        }

    def analyze_emotion(self, text: str) -> str:
        """Анализ эмоций в тексте"""
        # Простой анализ на основе ключевых слов
        emotions = {
            "happy": ["ха", "лол", "круто", "весело", "отлично"],
            "angry": ["блядь", "сука", "пиздец", "ебать", "бесит"],
            "sad": ["грустно", "печально", "хуёво", "херня"],
            "surprised": ["вау", "охуеть", "пиздец", "нереально"],
            "sarcastic": ["ага", "конечно", "да-да", "ясно"]
        }

        text_lower = text.lower()
        emotion_scores = {}

        for emotion, keywords in emotions.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            emotion_scores[emotion] = score

        return max(emotion_scores, key=emotion_scores.get) if any(
            emotion_scores.values()) else "neutral"

    def get_voice_style(self, emotion: str) -> Dict:
        """Настройки голоса для эмоции"""
        voice_styles = {
            "happy": {
                "pitch": 1.2,
                "speed": 1.1,
                "energy": 1.3
            },
            "angry": {
                "pitch": 0.9,
                "speed": 1.2,
                "energy": 1.5
            },
            "sad": {
                "pitch": 0.8,
                "speed": 0.9,
                "energy": 0.7
            },
            "surprised": {
                "pitch": 1.4,
                "speed": 1.0,
                "energy": 1.4
            },
            "sarcastic": {
                "pitch": 0.95,
                "speed": 0.95,
                "energy": 1.0
            },
            "neutral": {
                "pitch": 1.0,
                "speed": 1.0,
                "energy": 1.0
            }
        }
        return voice_styles.get(emotion, voice_styles["neutral"])

    def get_live2d_params(self, emotion: str) -> Dict:
        """Параметры Live2D для эмоции"""
        params = {
            "happy": {
                "mouth_open": 0.8,
                "eye_l_open": 1.0,
                "eye_r_open": 1.0,
                "eyebrow_l_y": 0.5,
                "eyebrow_r_y": 0.5,
                "mouth_form": 0.7  # улыбка
            },
            "angry": {
                "mouth_open": 0.3,
                "eye_l_open": 0.6,
                "eye_r_open": 0.6,
                "eyebrow_l_y": -0.5,
                "eyebrow_r_y": -0.5,
                "mouth_form": -0.5  # хмурый рот
            },
            "sad": {
                "mouth_open": 0.2,
                "eye_l_open": 0.4,
                "eye_r_open": 0.4,
                "eyebrow_l_y": -0.3,
                "eyebrow_r_y": -0.3,
                "mouth_form": -0.3
            },
            "surprised": {
                "mouth_open": 0.9,
                "eye_l_open": 1.5,
                "eye_r_open": 1.5,
                "eyebrow_l_y": 0.8,
                "eyebrow_r_y": 0.8,
                "mouth_form": 0.0
            }
        }
        return params.get(emotion, params["happy"])

    def search_internet(self, query: str) -> str:
        """Поиск информации в интернете"""
        try:
            # Пример с Google Custom Search API
            api_key = "YOUR_API_KEY"
            search_engine_id = "YOUR_SEARCH_ENGINE_ID"

            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": api_key,
                "cx": search_engine_id,
                "q": query,
                "num": 3
            }

            response = requests.get(url, params=params)
            results = response.json()

            # Формирование краткого ответа
            if "items" in results:
                info = []
                for item in results["items"][:2]:
                    info.append(f"{item['title']}: {item['snippet']}")
                return "\n".join(info)
            return "Не нашёл информации по этому запросу"

        except Exception as e:
            return f"Ошибка поиска: {str(e)}"

    def text_to_speech(self, text: str, emotion: str) -> bytes:
        """Преобразование текста в речь с эмоциями"""
        voice_style = self.get_voice_style(emotion)

        # Генерация речи с параметрами эмоций
        wav = self.tts_model.tts(
            text=text,
            speaker_wav="reference_voice.wav",  # Референсный голос
            language="ru",
            emotion=emotion,
            speed=voice_style["speed"])

        return wav

    def speech_to_text(self, audio_data: bytes) -> str:
        """Распознавание речи"""
        # Преобразование аудио данных
        audio_array = np.frombuffer(audio_data, dtype=np.float32)

        # Распознавание через Whisper
        result = self.stt_model.transcribe(audio_array)
        return result["text"]

    def update_live2d(self, params: Dict):
        """Обновление параметров Live2D модели"""
        if self.live2d_controller:
            self.live2d_controller.update_parameters(params)

    async def process_stream(self, websocket, path):
        """Обработка стрима в реальном времени"""
        context = []

        async for message in websocket:
            data = json.loads(message)

            if data["type"] == "text":
                user_input = data["content"]

                # Поиск в интернете если нужно
                if "найди" in user_input or "что такое" in user_input:
                    search_result = self.search_internet(user_input)
                    user_input += f"\n\nИнформация из поиска: {search_result}"

                # Генерация ответа
                response = self.generate_response(user_input, context)

                # Обновление контекста
                context.append(f"Пользователь: {user_input}")
                context.append(f"Стример: {response['text']}")

                # Обновление Live2D
                self.update_live2d(response['live2d_params'])

                # Генерация речи
                audio = self.text_to_speech(response['text'],
                                            response['emotion'])

                # Отправка ответа
                await websocket.send(
                    json.dumps({
                        "type": "response",
                        "text": response['text'],
                        "emotion": response['emotion'],
                        "audio": audio.tolist(),
                        "live2d_params": response['live2d_params']
                    }))

            elif data["type"] == "audio":
                # Обработка голосового сообщения
                audio_data = np.array(data["content"], dtype=np.float32)
                text = self.speech_to_text(audio_data)

                # Обработка как текстового сообщения
                await websocket.send(
                    json.dumps({
                        "type": "transcription",
                        "text": text
                    }))


class Live2DController:
    """Контроллер для Live2D модели"""

    def __init__(self):
        self.current_params = {}
        # Инициализация Live2D SDK

    def update_parameters(self, params: Dict):
        """Обновление параметров модели"""
        self.current_params.update(params)
        # Применение параметров к Live2D модели

    def animate_mouth_for_speech(self, audio_data: np.ndarray):
        """Анимация рта под речь"""
        # Анализ амплитуды звука для движения рта
        amplitude = np.abs(audio_data).mean()
        mouth_open = min(amplitude * 2, 1.0)

        self.update_parameters({"mouth_open": mouth_open})


# Запуск сервера
async def main():
    streamer = AIStreamer()
    streamer.load_models()

    # Запуск WebSocket сервера
    start_server = websockets.serve(streamer.process_stream, "localhost", 8765)

    await start_server
    print("AI Streamer запущен на порту 8765")
    await asyncio.Future()  # Работаем вечно


if __name__ == "__main__":
    asyncio.run(main())
