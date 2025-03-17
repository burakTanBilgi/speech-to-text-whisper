import os
import whisper

# Function to list files in a directory
def list_directory_contents(directory_path):
    print(f"Contents of directory '{directory_path}':")
    try:
        for item in os.listdir(directory_path):
            print(f"- {item}")
    except FileNotFoundError:
        print(f"Directory '{directory_path}' not found. Creating it now.")
        os.makedirs(directory_path)
    print()

# Function to perform speech-to-text conversion
def transcribe_audio(audio_file_path, model_name="base", language="tr"):
    # Load the Whisper model
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    # Transcribe the audio
    print(f"Transcribing file: {audio_file_path} in language: {language}")
    result = model.transcribe(audio_file_path, language=language)
    
    return result["text"]

# Main function
def main():
    # Specify the directory containing the audio file
    directory_path = "audio_files"
    
    # Show the contents of the directory
    list_directory_contents(directory_path)
    
    # Ask user for the audio file
    audio_filename = input("Enter the name of your audio file (or press Enter to use 'speech.mp3'): ")
    if not audio_filename:
        audio_filename = "speech.mp3"
    
    audio_file = os.path.join(directory_path, audio_filename)
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Warning: File '{audio_file}' does not exist.")
        print(f"Please place your audio file in the '{directory_path}' directory and run the script again.")
        return
    
    # Ask user for the model
    print("\nAvailable models: tiny, base, small, medium, large")
    print("Note: Larger models are more accurate but slower and require more RAM")
    model_name = input("Enter the model name (or press Enter to use 'base'): ")
    if not model_name:
        model_name = "base"
    
    # Ask user for the language
    print("\nLanguage codes examples: tr (Turkish), en (English), fr (French), de (German)")
    language = input("Enter the language code (or press Enter to use Turkish 'tr'): ")
    if not language:
        language = "tr"
    
    # Perform transcription
    transcription = transcribe_audio(audio_file, model_name, language)
    
    # Print and save the result
    print("\nTranscription result:")
    print(transcription)
    
    # Save to text file
    output_file = os.path.splitext(audio_filename)[0] + "_transcription.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcription)
    
    print(f"\nTranscription saved to {output_file}")

if __name__ == "__main__":
    main()