import asyncio
import websockets
import json
import pyaudio
import numpy as np
import threading
import queue


class StreamerClient:

    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.audio_queue = queue.Queue()
        self.is_recording = False

    async def connect(self):
        """Подключение к серверу"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            print("Подключено к AI-стримеру!")
            return True
        except Exception as e:
            print(f"Ошибка подключения: {e}")
            return False

    async def send_text_message(self, text: str):
        """Отправка текстового сообщения"""
        if not self.websocket:
            print("Не подключен к серверу")
            return

        message = {
            "type": "text",
            "content": text,
            "timestamp": asyncio.get_event_loop().time()
        }

        await self.websocket.send(json.dumps(message))
        print(f"Отправлено: {text}")

    async def send_audio_message(self, audio_data: np.ndarray):
        """Отправка аудио сообщения"""
        if not self.websocket:
            print("Не подключен к серверу")
            return

        message = {
            "type": "audio",
            "content": audio_data.tolist(),
            "timestamp": asyncio.get_event_loop().time()
        }

        await self.websocket.send(json.dumps(message))
        print("Отправлено аудио сообщение")

    async def listen_for_responses(self):
        """Прослушивание ответов от сервера"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self.handle_response(data)
        except websockets.exceptions.ConnectionClosed:
            print("Соединение закрыто")
        except Exception as e:
            print(f"Ошибка при получении ответа: {e}")

    async def handle_response(self, data):
        """Обработка ответа от стримера"""
        if data["type"] == "response":
            print(f"\n🎙️ Стример: {data['text']}")
            print(f"😊 Эмоция: {data['emotion']}")

            # Воспроизведение аудио (если есть)
            if "audio" in data:
                audio_data = np.array(data["audio"], dtype=np.float32)
                self.play_audio(audio_data)

            # Обновление Live2D параметров
            if "live2d_params" in data:
                print(f"🎭 Live2D: {data['live2d_params']}")

        elif data["type"] == "transcription":
            print(f"🎤 Распознано: {data['text']}")

    def play_audio(self, audio_data: np.ndarray):
        """Воспроизведение аудио"""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=22050,
                            output=True)
            stream.write(audio_data.tobytes())
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            print(f"Ошибка воспроизведения аудио: {e}")

    def start_voice_recording(self):
        """Начало записи голоса"""
        self.is_recording = True
        threading.Thread(target=self._record_audio, daemon=True).start()

    def stop_voice_recording(self):
        """Остановка записи голоса"""
        self.is_recording = False

    def _record_audio(self):
        """Запись аудио в отдельном потоке"""
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024)

        print("🎤 Запись началась...")
        frames = []

        while self.is_recording:
            data = stream.read(1024)
            frames.append(np.frombuffer(data, dtype=np.float32))

        stream.stop_stream()
        stream.close()
        p.terminate()

        if frames:
            audio_data = np.concatenate(frames)
            self.audio_queue.put(audio_data)
            print("🎤 Запись завершена")

    async def interactive_chat(self):
        """Интерактивный чат с стримером"""
        print("\n=== AI Стример Чат ===")
        print("Команды:")
        print("  /voice - запись голосового сообщения")
        print("  /quit - выход")
        print("  Любой текст - отправка текстового сообщения")
        print("========================\n")

        while True:
            try:
                user_input = input("\nВы: ").strip()

                if user_input == "/quit":
                    break
                elif user_input == "/voice":
                    print("Нажмите Enter для начала записи...")
                    input()
                    self.start_voice_recording()

                    print("Говорите... (Enter для остановки)")
                    input()
                    self.stop_voice_recording()

                    # Ждем завершения записи
                    await asyncio.sleep(0.5)

                    if not self.audio_queue.empty():
                        audio_data = self.audio_queue.get()
                        await self.send_audio_message(audio_data)

                elif user_input:
                    await self.send_text_message(user_input)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Ошибка: {e}")

    async def run(self):
        """Запуск клиента"""
        if not await self.connect():
            return

        # Запуск прослушивания в фоне
        listen_task = asyncio.create_task(self.listen_for_responses())

        # Запуск интерактивного чата
        chat_task = asyncio.create_task(self.interactive_chat())

        # Ожидание завершения любой из задач
        done, pending = await asyncio.wait([listen_task, chat_task],
                                           return_when=asyncio.FIRST_COMPLETED)

        # Отмена оставшихся задач
        for task in pending:
            task.cancel()

        if self.websocket:
            await self.websocket.close()

        print("Отключено от стримера")


# Простой клиент для быстрого тестирования
class SimpleClient:

    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url

    async def send_message(self, text: str):
        """Отправка одного сообщения"""
        async with websockets.connect(self.server_url) as websocket:
            message = {
                "type": "text",
                "content": text,
                "timestamp": asyncio.get_event_loop().time()
            }

            await websocket.send(json.dumps(message))

            # Ожидание ответа
            response = await websocket.recv()
            data = json.loads(response)

            if data["type"] == "response":
                print(f"Стример: {data['text']}")
                print(f"Эмоция: {data['emotion']}")
                return data

    async def quick_test(self):
        """Быстрый тест"""
        messages = [
            "Привет! Как дела?", "Расскажи шутку", "Банан или огурец?"
            # "Найди информацию о последних новостях игровой индустрии"
        ]

        for msg in messages:
            print(f"\nВы: {msg}")
            try:
                await self.send_message(msg)
                await asyncio.sleep(2)  # Пауза между сообщениями
            except Exception as e:
                print(f"Ошибка: {e}")


# Запуск клиента
async def main():
    print("Выберите режим:")
    print("1. Интерактивный чат")
    print("2. Быстрый тест")

    choice = input("Ваш выбор (1/2): ").strip()

    if choice == "1":
        client = StreamerClient()
        await client.run()
    elif choice == "2":
        client = SimpleClient()
        await client.quick_test()
    else:
        print("Неверный выбор")


if __name__ == "__main__":
    asyncio.run(main())
