🚀 Step-by-Step Guide to Run the Chatbot Project
🖥️ Backend (FastAPI Server)

    Navigate to the Chatbot Folder

        Open a terminal or command prompt.

        Use the command:

    cd path/to/Chatbot

    Replace path/to/Chatbot with the actual folder location.

Set Up the Python Virtual Environment (First-Time Setup Only)

    Run the following command to create a virtual environment:

    python -m venv chatbot-env

    This will create a chatbot-env folder inside Chatbot.

Activate the Virtual Environment

    Inside the Chatbot folder, run:

    chatbot-env\Scripts\activate

    This ensures that dependencies are correctly managed.

Install Required Dependencies

    Run:

    pip install -r requirements.txt

    This installs all necessary packages.

Start the FastAPI Server

    Run:

python main.py

Wait for: Application startup complete message.

The server is now running and listening for WebSocket connections.
