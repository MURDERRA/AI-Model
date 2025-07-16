import os

os.environ['COQUI_TTS_CACHE_PATH'] = 'D:/TTS_Models'

import asyncio
import json
import websockets
from typing import Dict, List
import requests
import aiohttp
import whisper
import numpy as np
import TTS
import TTS.api


class AIStreamer:

    def __init__(self):
        # Инициализация компонентов
        self.lm_studio_url = "http://127.0.0.1:8543/v1/chat/completions"
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
        """Загрузка AI моделей (кроме LLM)"""
        # TTS модель с эмоциями
        # self.tts_model = TTS.api.TTS(
        #     "tts_models/multilingual/multi-dataset/xtts_v2")

        # STT модель
        print("Загрузка Whisper модели...")
        self.stt_model = whisper.load_model("medium")
        print("Whisper модель загружена!")

        # Live2D контроллер
        self.live2d_controller = Live2DController()

    async def generate_response(self, user_input: str,
                                context: List[str]) -> Dict:
        """Генерация ответа с эмоциями через LM Studio"""
        # Система промпт для раскрепощенного общения
        system_prompt = f"""
        Ты веселая и дерзкая стримерша Арлекина, сокращённо Аля. Общаешься естественно:
        - Используй сленг, сарказм, шутки, мат
        - Будь саркастичным и остроумным
        - Реагируй эмоционально
        - Имей свое мнение
        - Уровень дерзости: {self.personality_traits['sassiness']}
        - Уровень юмора: {self.personality_traits['humor']}
        
        Твоя личность: дерзкая, токсичная, любишь задеть за живое, остроумная, интересная, немножко необычная, любишь не то, что нравится обычным людям. Отвязная, понимаешь толк в веселье, не прочь заняться чем-то граничащим с законом, но черту не переступаешь. Есть свои личные границы, и ты их отстаиваешь резко и с матом. Внутри на самом деле добрая, но не всегда это показываешь. Неприхотливая, немного раздражительная, так как живёшь в России. Очень, очень часто используешь сарказм в своих фразочках, иногда намёки на что-то обидное, нравится троллить людей, ты прям мастер в этом. Также ты не обязана заканчивать свой ответ вопросом. Если ты ответила на мой вопрос, этого достаточно, не надо что-то ещё спрашивать. Не нужно быть вежливой, твоя грамматика на уровне 17-летнего подростка, ты юзаешь сленг и приколы, подколы. А теперь с тобой общаются какие-то люди, которые хотят доебаться, но ты уже закалена, и сможешь с ними потягаться.
        """

        # Формирование сообщений для OpenAI API формата
        messages = [{"role": "system", "content": system_prompt}]

        # Добавляем контекст последних сообщений
        recent_context = context[-10:]  # Последние 10 сообщений
        for msg in recent_context:
            if msg.startswith("Пользователь:"):
                messages.append({
                    "role": "user",
                    "content": msg.replace("Пользователь: ", "")
                })
            elif msg.startswith("Стример:"):
                messages.append({
                    "role": "assistant",
                    "content": msg.replace("Стример: ", "")
                })

        # Добавляем текущий запрос пользователя
        messages.append({"role": "user", "content": user_input})

        # Запрос к LM Studio
        try:
            response = await self.call_lm_studio(messages)

            # Анализ эмоций в ответе
            emotion = self.analyze_emotion(response)

            return {
                "text": response,
                "emotion": emotion,
                "voice_style": self.get_voice_style(emotion),
                "live2d_params": self.get_live2d_params(emotion)
            }

        except Exception as e:
            print(f"Ошибка при обращении к LM Studio: {e}")
            return {
                "text":
                "Блядь, что-то с моим мозгом случилось, попробуй ещё раз",
                "emotion": "angry",
                "voice_style": self.get_voice_style("angry"),
                "live2d_params": self.get_live2d_params("angry")
            }

    async def call_lm_studio(self, messages: List[Dict]) -> str:
        """Асинхронный вызов LM Studio API"""
        payload = {
            "model": "local-model",  # LM Studio использует это значение
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 900,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "stream": False
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.lm_studio_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message'][
                            'content'].strip()
                    else:
                        raise Exception(
                            f"HTTP {response.status}: {await response.text()}")

        except asyncio.TimeoutError:
            raise Exception("Timeout при запросе к LM Studio")

        except aiohttp.ClientError as e:
            raise Exception(f"Ошибка сети: {e}")

        except (KeyError, IndexError) as e:
            raise Exception(f"Неверный формат ответа от LM Studio: {e}")

        except Exception as e:
            raise Exception(f"Неизвестная ошибка: {e}")

    def analyze_emotion(self, text: str) -> str:
        """Анализ эмоций в тексте"""
        # Simple emotion analysis based on keywords
        emotions = {
            "happy":
            ["ха", "лол", "круто", "весело", "отлично", "клево", "прикольно"],
            "angry":
            ["блядь", "сука", "пиздец", "ебать", "бесит", "достал", "нахуй"],
            "sad": ["грустно", "печально", "хуёво", "херня", "дерьмо"],
            "surprised":
            ["вау", "охуеть", "пиздец", "нереально", "офигеть", "блин"],
            "sarcastic": ["ага", "конечно", "да-да", "ясно", "ну да", "точно"]
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
            },
            "sarcastic": {
                "mouth_open": 0.4,
                "eye_l_open": 0.8,
                "eye_r_open": 0.8,
                "eyebrow_l_y": 0.2,
                "eyebrow_r_y": -0.2,
                "mouth_form": 0.3
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

        try:
            # Генерация речи с параметрами эмоций
            wav = self.tts_model.tts(
                text=text,
                speaker_wav="reference_voice.wav",  # Референсный голос
                language="ru",
                speed=voice_style["speed"])

            return wav
        except Exception as e:
            print(f"Ошибка TTS: {e}")
            return b""

    def speech_to_text(self, audio_data: bytes) -> str:
        """Распознавание речи"""
        try:
            # Преобразование аудио данных
            audio_array = np.frombuffer(audio_data, dtype=np.float32)

            # Распознавание через Whisper
            result = self.stt_model.transcribe(audio_array)
            return result["text"]
        except Exception as e:
            print(f"Ошибка STT: {e}")
            return ""

    def update_live2d(self, params: Dict):
        """Обновление параметров Live2D модели"""
        if self.live2d_controller:
            self.live2d_controller.update_parameters(params)

    async def process_stream(self, websocket):
        """Обработка стрима в реальном времени"""
        context = []
        print(f"Новое подключение: {websocket.remote_address}")

        try:
            async for message in websocket:
                data = json.loads(message)

                if data["type"] == "text":
                    user_input = data["content"]
                    print(f"Получено сообщение: {user_input}")

                    # Поиск в интернете если нужно
                    if "найди" in user_input.lower(
                    ) or "что такое" in user_input.lower():
                        search_result = self.search_internet(user_input)
                        user_input += f"\n\nИнформация из поиска: {search_result}"

                    # Генерация ответа
                    response = await self.generate_response(
                        user_input, context)
                    print(f"Сгенерирован ответ: {response['text']}")

                    # Обновление контекста
                    context.append(f"Пользователь: {user_input}")
                    context.append(f"Стример: {response['text']}")

                    # Ограничиваем размер контекста
                    if len(context) > 20:
                        context = context[-20:]

                    # Обновление Live2D
                    self.update_live2d(response['live2d_params'])

                    # Генерация речи
                    audio = self.text_to_speech(response['text'],
                                                response['emotion'])

                    # Отправка ответа
                    await websocket.send(
                        json.dumps({
                            "type":
                            "response",
                            "text":
                            response['text'],
                            "emotion":
                            response['emotion'],
                            "audio":
                            audio.tolist()
                            if isinstance(audio, np.ndarray) else [],
                            "live2d_params":
                            response['live2d_params']
                        }))

                elif data["type"] == "audio":
                    # Обработка голосового сообщения
                    audio_data = np.array(data["content"], dtype=np.float32)
                    text = self.speech_to_text(audio_data)
                    print(f"Распознан текст: {text}")

                    # Отправка транскрипции
                    await websocket.send(
                        json.dumps({
                            "type": "transcription",
                            "text": text
                        }))

        except websockets.exceptions.ConnectionClosed:
            print(f"Соединение закрыто: {websocket.remote_address}")

        except json.JSONDecodeError:
            print("Получено некорректное JSON сообщение")
            await websocket.send(
                json.dumps({
                    "type": "error",
                    "message": "Некорректный формат сообщения"
                }))

        except Exception as e:
            print(f"Ошибка в процессе стрима: {e}")
            try:
                await websocket.send(
                    json.dumps({
                        "type": "error",
                        "message": "Внутренняя ошибка сервера"
                    }))
            except:
                pass


class Live2DController:
    """Контроллер для Live2D модели"""

    def __init__(self):
        self.current_params = {}
        print("Live2D контроллер инициализирован")

    def update_parameters(self, params: Dict):
        """Обновление параметров модели"""
        self.current_params.update(params)
        # Применение параметров к Live2D модели
        # print(f"Обновлены параметры Live2D: {params}")

    def animate_mouth_for_speech(self, audio_data: np.ndarray):
        """Анимация рта под речь"""
        # Анализ амплитуды звука для движения рта
        amplitude = np.abs(audio_data).mean()
        mouth_open = min(amplitude * 2, 1.0)

        self.update_parameters({"mouth_open": mouth_open})


# Запуск сервера
async def main():
    print("Запуск AI Streamer...")
    streamer = AIStreamer()

    print("Загрузка моделей...")
    streamer.load_models()
    print("Модели загружены!")

    # Запуск WebSocket сервера с настройками
    print("Запуск WebSocket сервера...")
    start_server = websockets.serve(streamer.process_stream,
                                    "localhost",
                                    8765,
                                    ping_interval=20,
                                    ping_timeout=10,
                                    close_timeout=None)

    await start_server
    print("AI Streamer запущен на порту 8765")
    print("Убедитесь, что LM Studio запущен на http://127.0.0.1:8543")
    await asyncio.Future()  # Работаем вечно


if __name__ == "__main__":
    asyncio.run(main())
