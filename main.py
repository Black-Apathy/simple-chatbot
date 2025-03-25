import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from chatbot import ChatBot

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create a ChatBot instance
chatbot = ChatBot()


def respond(query):
    answer = chatbot.find_best_answer(query)
    return answer


# Root endpoint (for testing)
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


# Initialize the server status
server_status = "initializing"


@app.get("/status")
async def get_status():
    global server_status
    return {"status": server_status}


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.debug("✅ WebSocket connection established.")

    try:
        while True:
            try:
                # Wait for client message (max wait time: 30 sec)
                query = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                logging.debug(f"📩 Received from client: {query}")

                # Simulated response
                if query != "ping":  # Ignore ping messages from client
                    response = f"🤖 AI: {respond(query)}"
                    await websocket.send_text(response)
                    logging.debug(f"📤 Sent to client: {response}")

            except asyncio.TimeoutError:
                await websocket.send_text("ping")  # Keep WebSocket alive
                logging.debug("🔄 Sent heartbeat (ping) to client")

    except WebSocketDisconnect:
        logging.debug("❌ Client disconnected.")
    except Exception as e:
        logging.error(f"⚠️ WebSocket error: {e}")
    finally:
        logging.debug("🔌 Closing WebSocket connection.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
