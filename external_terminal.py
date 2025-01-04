import requests
from colorama import Fore, Style, init

# Initialize Colorama
init()

def chat_with_bot():
    print(f"{Fore.YELLOW}Welcome to the Health Care Chatbot!{Style.RESET_ALL}")
    print(f"Type '{Fore.RED}exit{Style.RESET_ALL}' to end the conversation.\n")
    
    while True:
        # Get user input
        user_message = input(f"{Fore.GREEN}You:{Style.RESET_ALL} ")
        if user_message.lower() == "exit":
            print(f"{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
            break
        
        # Send message to Flask server
        try:
            response = requests.post(
                "http://127.0.0.1:5000/chat",  # URL of your Flask app
                json={"message": user_message}  # Send message as JSON
            )
            if response.status_code == 200:
                bot_response = response.json().get("response", "I'm sorry, I couldn't process that.")
                print(f"{Fore.BLUE}Bot:{Style.RESET_ALL} {bot_response}")
            else:
                print(f"{Fore.RED}Bot: An error occurred on the server.{Style.RESET_ALL}")
        except requests.exceptions.RequestException as e:
            print(f"{Fore.RED}Error connecting to the bot server: {e}{Style.RESET_ALL}")
            break

if __name__ == "__main__":
    chat_with_bot()
