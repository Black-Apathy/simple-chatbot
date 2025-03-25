import asyncio
import websockets
import requests
import aioconsole


async def test_websocket():
    uri = "ws://127.0.0.1:8000/ws"
    print("Connecting to the server...")

    while True:
        try:
            response = requests.get("http://127.0.0.1:8000/status")
            if response.status_code == 200:
                print("Server is ready. You can start chatting now.")
                break
            print("Server is initializing... Please wait.")
            await asyncio.sleep(1)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            await asyncio.sleep(1)

    async with websockets.connect(uri) as websocket:
        print('Please press "exit" to close the chat!')
        while True:
            try:
                # Get input from user
                message = await aioconsole.ainput("You: ")
                if message.lower() == "exit":
                    print("Exiting chat, closing connection.")
                    # Send an exit message to the server before breaking
                    await websocket.send(message)
                    # Receive the final goodbye message
                    response = await websocket.recv()
                    print(f"Server: {response}")
                    break  # Exit the loop and disconnect cleanly

                # Send the message to the server
                await websocket.send(message)

                # Receive the response
                response = await websocket.recv()
                print(f"Server: {response}")
            except websockets.ConnectionClosed:
                print("Connection closed.")
                break


async def main():
    try:
        await test_websocket()
    except Exception as e:
        print(f"An error occurred: {e}")


asyncio.run(main())
