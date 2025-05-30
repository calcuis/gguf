import os
def wav_handler():
    wav_files = [file for file in os.listdir() if file.endswith('.wav')]
    if wav_files:
        print('WAV file(s) available. Select which one to use:')
        for index, file_name in enumerate(wav_files, start=1):
            print(f'{index}. {file_name}')
        choice = input(f'Enter your choice (1 to {len(wav_files)}): ')
        try:
            choice_index = int(choice) - 1
            selected_file = wav_files[choice_index]
            print(f'WAV file: {selected_file} is selected!')
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(selected_file) as source:
                audio = r.record(source)
                output = r.recognize_google(audio)
            try:
                from rich.console import Console
                console = Console()
                console.print(f'Content recognized as: [green]' + output)
                input(
                    '\n---Enter to prompt the WAV content recognized above into GGUF model---'
                    )
                print('Processing...')
                ans = llm(output)
                print(output + ans)
            except sr.UnknownValueError:
                print('Could not understand audio')
            except sr.RequestError as e:
                print('Error; {0}'.format(e))
        except (ValueError, IndexError):
            print('Invalid choice. Please enter a valid number.')
    else:
        print('No WAV files are available in the current directory.')
        input('--- Press ENTER To Skip ---')
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]
if gguf_files:
    print('GGUF file(s) available. Select which one to use:')
    for index, file_name in enumerate(gguf_files, start=1):
        print(f'{index}. {file_name}')
    choice = input(f'Enter your choice (1 to {len(gguf_files)}): ')
    try:
        choice_index = int(choice) - 1
        selected_file = gguf_files[choice_index]
        print(f'Model file: {selected_file} is selected!')
        ModelPath = selected_file
        print('Processing...')
        from ctransformers import AutoModelForCausalLM
        llm = AutoModelForCausalLM.from_pretrained(ModelPath)
        while True:
            ask = input('---Enter to select a WAV file (Q for quit)---')
            if ask == 'q' or ask == 'Q':
                break
            wav_handler()
    except (ValueError, IndexError):
        print('Invalid choice. Please enter a valid number.')
else:
    print('No GGUF files are available in the current directory.')
    input('--- Press ENTER To Exit ---')
print('Goodbye!')