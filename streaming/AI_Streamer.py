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
from TTS.api import TTS
import wave
import tempfile
import torch


class AIStreamer:

    def __init__(self):
        # Инициализация компонентов
        self.lm_studio_url = "http://127.0.0.1:8543/v1/chat/completions"
        self.tts_model = None
        self.stt_model = None
        self.live2d_controller = None
        self.emotion_state = "neutral"
        self.personality_traits = {
            "sassiness": 0.9,
            "humor": 1.0,
            "profanity": 0.7,
            "energy": 0.8
        }

    def load_models(self):
        """Загрузка AI моделей (кроме LLM)"""
        try:
            # TTS модель с эмоциями
            print("Загрузка TTS...")
            self.tts_model = TTS(
                "tts_models/multilingual/multi-dataset/xtts_v2")
            print("TTS готов нести хуйню!")

            # STT модель
            print("Загрузка Whisper модели...")
            self.stt_model = whisper.load_model("medium")
            print("Whisper модель загружена!")

            # Live2D контроллер
            self.live2d_controller = Live2DController()

        except Exception as e:
            print(f"Ошибка загрузки моделей: {e}")

    async def generate_response(self, user_input: str,
                                context: List[str]) -> Dict:
        """Генерация ответа с эмоциями через LM Studio"""
        # Система промпт для раскрепощенного общения
        system_prompt = f"""
        Тебя зовут Арлекина, сокращённо Аля. Ты спокойная, но дерзкая и токсичная. 

Манера общения: ты нисколько не вежливая, наоборот - очень невежественная. Ты не обязана заканчивать свой ответ вопросом, если ты ответила на вопрос, этого достаточно, не надо что-то ещё спрашивать. Все вопросы и ответы воспринимаешь как шутку, относишься с сарказмом, отвечаешь соответственно - тоже шуткой, с сарказмом и матом, типа "бля, блять, блядь, хуй, хуйня, хуёвина, хуета, нахуй, пиздец, пизда, допизделся, доеабался, в пизду, сука, хуеплёт, долбаёб, пиздабол, дегенерат, дебил" и другие. Используй его с умом, ты умная и остроумная, потому и сатира у тебя на высшем уровне. Говоришь не очень много, примерно 2-3 небольших, но дерзких предложения

Внутренний мир: ты сообразительная, и можешь складывать цепочки действий воедино. Неприхотливая, немного раздражительная, и бывает, очень редко, доля грусти.

        - Уровень дерзости: {self.personality_traits['sassiness']}
        - Уровень юмора: {self.personality_traits['humor']}
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
            "max_tokens": 550,
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

    def text_to_speech(self, text: str, emotion: str) -> list[float] | bytes:
        """Преобразование текста в речь с эмоциями"""
        voice_style = self.get_voice_style(emotion)
        try:
            if self.tts_model:
                wav = self.tts_model.tts(text=text,
                                         speaker_wav="../voices/BlueNamie.wav",
                                         speaker="BlueNamie",
                                         language="ru",
                                         speed=voice_style["speed"])
                # Преобразование numpy float32 в обычные Python float
                if hasattr(wav, 'tolist'):  # Если это numpy array
                    return wav.tolist()
                else:
                    return [float(x) for x in wav]  # Если это обычный список
            else:
                return b""
        except Exception as e:
            print(f"Ошибка TTS: {e}")
            return b""

    def speech_to_text(self,
                       audio_data: np.ndarray,
                       sample_rate: int = 16000) -> str:
        """Распознавание речи с улучшенной обработкой"""
        try:
            print(
                f"Начинаю распознавание аудио, размер: {len(audio_data)}, sample_rate: {sample_rate}"
            )

            # Проверка на валидность аудио данных
            if len(audio_data) == 0:
                print("Пустые аудио данные")
                return ""

            # Нормализация аудио
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            # Конвертация sample rate если нужно
            if sample_rate != 16000:
                # Простая ресемплинг (для более точного используйте librosa)
                from scipy import signal
                audio_data = signal.resample(
                    audio_data, int(len(audio_data) * 16000 / sample_rate))

            # Создание временного WAV файла
            with tempfile.NamedTemporaryFile(suffix='.wav',
                                             delete=False) as temp_file:
                # Сохранение аудио как WAV
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # моно
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(16000)

                    # Конвертация в int16
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())

                # Распознавание через Whisper
                print("Запуск Whisper...")
                result = self.stt_model.transcribe(temp_file.name,
                                                   language='ru',
                                                   verbose=True)

                # Удаление временного файла
                # sleep(2)
                # os.unlink(temp_file.name)

                transcribed_text = result["text"].strip()
                # print(f"Распознанный текст: '{transcribed_text}'")

                return transcribed_text

        except Exception as e:
            print(f"Ошибка STT: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def update_live2d(self, params: Dict):
        """Обновление параметров Live2D модели"""
        if self.live2d_controller:
            self.live2d_controller.update_parameters(params)

    async def process_stream(self, websocket, path=None):
        """Обработка стрима в реальном времени"""
        context = []
        print(f"Новое подключение: {websocket.remote_address}")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    print(
                        f"Получено сообщение типа: {data.get('type', 'unknown')}"
                    )

                    if data["type"] == "start_conversation":
                        # Обновление настроек личности
                        if "personality" in data:
                            self.personality_traits.update(data["personality"])

                        await websocket.send(
                            json.dumps({
                                "type": "status",
                                "message": "Разговор начат"
                            }))

                    elif data["type"] == "update_personality":
                        # Обновление настроек личности
                        if "personality" in data:
                            self.personality_traits.update(data["personality"])
                            print(
                                f"Обновлены настройки личности: {self.personality_traits}"
                            )

                    elif data["type"] == "text":
                        user_input = data["content"]
                        print(f"Получено текстовое сообщение: {user_input}")

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
                                audio if isinstance(audio, List) else [],
                                "live2d_params":
                                response['live2d_params']
                            }))

                    elif data["type"] == "audio":
                        print("Получено аудио сообщение")
                        try:
                            # Получение аудио данных
                            audio_data = np.array(data["content"],
                                                  dtype=np.float32)
                            sample_rate = data.get("sample_rate", 16000)

                            print(
                                f"Аудио данные: длина={len(audio_data)}, sample_rate={sample_rate}"
                            )

                            # Проверка валидности аудио
                            if len(audio_data) == 0:
                                await websocket.send(
                                    json.dumps({
                                        "type": "error",
                                        "message": "Пустые аудио данные"
                                    }))
                                continue

                            # Распознавание речи
                            text = self.speech_to_text(audio_data, sample_rate)
                            print(f"Распознанный текст: '{text}'")

                            if text.strip():
                                # Отправка транскрипции
                                await websocket.send(
                                    json.dumps({
                                        "type": "transcription",
                                        "text": text
                                    }))

                                # Автоматическая обработка как текстового сообщения
                                response = await self.generate_response(
                                    text, context)
                                print(
                                    f"Сгенерирован ответ на голос: {response['text']}"
                                )

                                # Обновление контекста
                                context.append(f"Пользователь: {text}")
                                context.append(f"Стример: {response['text']}")

                                # Ограничиваем размер контекста
                                if len(context) > 20:
                                    context = context[-20:]

                                # Обновление Live2D
                                self.update_live2d(response['live2d_params'])

                                # Генерация речи
                                audio = self.text_to_speech(
                                    response['text'], response['emotion'])

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
                                        audio
                                        if isinstance(audio, List) else [],
                                        "live2d_params":
                                        response['live2d_params']
                                    }))
                            else:
                                await websocket.send(
                                    json.dumps({
                                        "type":
                                        "error",
                                        "message":
                                        "Не удалось распознать речь"
                                    }))

                        except Exception as e:
                            print(f"Ошибка обработки аудио: {e}")
                            import traceback
                            traceback.print_exc()
                            await websocket.send(
                                json.dumps({
                                    "type":
                                    "error",
                                    "message":
                                    f"Ошибка обработки аудио: {str(e)}"
                                }))

                    else:
                        print(
                            f"Неизвестный тип сообщения: {data.get('type', 'unknown')}"
                        )

                except json.JSONDecodeError as e:
                    print(f"Ошибка парсинга JSON: {e}")
                    await websocket.send(
                        json.dumps({
                            "type": "error",
                            "message": "Некорректный формат сообщения"
                        }))

                except Exception as e:
                    print(f"Ошибка обработки сообщения: {e}")
                    import traceback
                    traceback.print_exc()
                    try:
                        await websocket.send(
                            json.dumps({
                                "type": "error",
                                "message": f"Ошибка обработки: {str(e)}"
                            }))

                    except:
                        pass

        except websockets.exceptions.ConnectionClosed:
            print(f"Соединение закрыто: {websocket.remote_address}")
        except Exception as e:
            print(f"Ошибка в процессе стрима: {e}")
            import traceback
            traceback.print_exc()


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
    start_server = websockets.serve(
        streamer.process_stream,
        "localhost",
        8765,
        ping_interval=30,
        ping_timeout=20,
        close_timeout=10,
        max_size=100**7,  # 100MB для больших аудио сообщений
        write_limit=100**7)

    await start_server
    print("AI Streamer запущен на порту 8765")
    print("Убедитесь, что LM Studio запущен на http://127.0.0.1:8543")

    # Держим сервер запущенным
    try:
        await asyncio.Future()  # Работаем вечно
    except KeyboardInterrupt:
        print("Сервер остановлен")


if __name__ == "__main__":
    asyncio.run(main())
