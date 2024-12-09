import matplotlib
import magic
matplotlib.use('Agg')  # Fix for GUI backend error
import telebot
import requests
import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
API_TOKEN = '7214725097:AAEXF5LWWlXS60OFL3w2PLOgDhPmyJCGApc'
SETTINGS = {
    'N_FFT': 8820//2,   # Default FFT window size
    'HOP_LEN':  8820//4,  # Default hop length
    'WIN_LEN':  8820//2, #default win lenght
    'MAGNITUDE_AMP_MULTIPLIER':10.0, #by how much increase the amplituted of DTMF freqs before applying threshold filtering
    'THRESHOLD_PERCENTILE': 96.0, #which percentile to use as a threshold
    'MASKS_MEDIAN_MULTIPLIER':10.0, #by how much to multiply median when extractin keys
    'PLT_YLIM_LOWER':0, #lower freq to show
    'PLT_YLIM_UPPER':16384, #upper freq to show
    'FIG_SIZE_X':10, # control width of diagram
    'FIG_SIZE_Y':4, # control height of diagram
}


bot = telebot.TeleBot(API_TOKEN)


def fetch_audio_as_buffer(file_id):
    file_info = bot.get_file(file_id)
    file_url = f"https://api.telegram.org/file/bot{API_TOKEN}/{file_info.file_path}"
    response = requests.get(file_url)
    response.raise_for_status()
    return io.BytesIO(response.content)

# Handle '/start' and '/help'
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, """\
Hello Agent! I am your tool to analyse DTMF frequencies!\
""")

@bot.message_handler(commands=['help'])
def send_welcome(message):
    bot.reply_to(message, """\
Just record me an audio message or audio, and I will give you keys pressed!\
""")


@bot.message_handler(content_types=['audio', 'voice','document'])
def handle_audio_message(message):
    print(f"Received content type: {message.content_type}")
    try:
        audio_buffer = None

        # Handle audio and voice messages
        if message.content_type in ['audio', 'voice']:
            file_id = message.audio.file_id if message.content_type == 'audio' else message.voice.file_id
            audio_buffer = fetch_audio_as_buffer(file_id)

        # Handle document messages for WAV files
        elif message.content_type == 'document':
            file_id = message.document.file_id
            audio_buffer = fetch_audio_as_buffer(file_id)
            file_info = bot.get_file(message.document.file_id)
            if not file_info.file_path.endswith('.wav'):
                bot.reply_to(message, "Please send a valid WAV file with the .wav extension.")
                return

        else:
            bot.reply_to(message, "Unsupported file type. Please send audio or WAV files.")
            return

        if audio_buffer is None:
            bot.reply_to(message, "An error occurred: Could not process the file.")
            return
        bot.reply_to(message, "Started tha analisys of audio",parse_mode="MarkdownV2")

        
        # Fetch audio file as a buffer
        audio_buffer.seek(0)  # Ensure the buffer is at the beginning
        y, sr = librosa.load(audio_buffer,sr=None)

        print(sr)
    # If audio is stereo, convert it to mono
        if len(y.shape) > 1:
            y = librosa.to_mono(y.T)


        stft_result = librosa.stft(y,n_fft=SETTINGS['N_FFT'],  win_length=SETTINGS['WIN_LEN'],hop_length=SETTINGS['HOP_LEN'], window='hann', center=True, dtype=None, pad_mode='constant', out=None)
        magnitude,phase = librosa.magphase(stft_result)
        spectrogram = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
    
        # Generate spectrogram image
        print("Plotting")
        plt.figure(figsize=(SETTINGS['FIG_SIZE_X'], SETTINGS['FIG_SIZE_Y']))
        librosa.display.specshow(spectrogram,hop_length=SETTINGS['HOP_LEN'], sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.ylim(SETTINGS['PLT_YLIM_LOWER'],SETTINGS['PLT_YLIM_UPPER'])
        # Save the plot to an in-memory buffer
        image_buffer = io.BytesIO()
        plt.savefig(image_buffer, format='png')
        plt.close()  # Close the plot to free up memory
        image_buffer.seek(0)
        # Send the spectrogram image back to the user
        bot.send_photo(message.chat.id, image_buffer, caption="Here is the spectrogram of initial audio")
        
        # threshold filtering

        frequencies = librosa.fft_frequencies(sr=sr, n_fft=SETTINGS['N_FFT'])
        freq_range = (frequencies >= 691) & (frequencies <= 1640)  # Boolean mask for frequency range corresponding to DTMF range
        magnitude_in_range = magnitude[freq_range, :]  # Extract magnitudes in this range
        n = SETTINGS['THRESHOLD_PERCENTILE'] # Define the percentile
        percentile_magnitude = np.percentile(magnitude_in_range, n)
        threshold = percentile_magnitude
        #apmplifying the DTMF freqs
        dtmf_frequencies = [697, 770, 852, 941, 1209, 1336, 1477, 1633]
        # Get the frequency bins corresponding to DTMF tones
        dtmf_bins = [np.argmin(np.abs(frequencies - freq)) for freq in dtmf_frequencies]
        #Amplify the DTMF frequencies by factor
        amplified_magnitude = np.copy(magnitude)
        for bin_idx in dtmf_bins:
            amplified_magnitude[bin_idx, :] *= SETTINGS['MAGNITUDE_AMP_MULTIPLIER'] #tweaking
        print(threshold)
        stft_filtered = amplified_magnitude * (np.abs(amplified_magnitude) > threshold) * phase


        print("PLOTTING 2")
        plt.figure(figsize=(SETTINGS['FIG_SIZE_X'], SETTINGS['FIG_SIZE_Y']))
        librosa.display.specshow(librosa.amplitude_to_db(abs(stft_filtered), ref=np.max),hop_length=SETTINGS['HOP_LEN'], sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.ylim(SETTINGS['PLT_YLIM_LOWER'],SETTINGS['PLT_YLIM_UPPER'])
        #plt.xlim(SETTINGS['PLT_XLIM_LOWER'],SETTINGS['PLT_XLIM_UPPER'])
        # Save the plot to an in-memory buffer
        image_buffer = io.BytesIO()
        plt.savefig(image_buffer, format='png')
        plt.close()  # Close the plot to free up memory
        image_buffer.seek(0)# Send the spectrogram image back to the user

        bot.send_photo(message.chat.id, image_buffer, caption="Here is the spectrogram of cleaned audio")
        bot.reply_to(message, "Calculating the masks median",parse_mode="MarkdownV2")
        
        dtmf_frequencies = {
            '1': [697, 1209],
            '2': [697, 1336],
            '3': [697, 1477],
            '4': [770, 1209],
            '5': [770, 1336],
            '6': [770, 1477],
            '7': [852, 1209],
            '8': [852, 1336],
            '9': [852, 1477],
            '0': [941, 1336],
            '*': [941, 1209],
            '#': [941, 1477],
            'A':[697,1633],
            'B':[770,1633],
            'C':[852,1633],
            'D':[941,1633]
        }

        # Get the frequencies corresponding to STFT rows
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=SETTINGS['N_FFT'])

        #Generate masks for each key
        masks = {}
        for key, freqs in dtmf_frequencies.items():
            mask = np.zeros_like(frequencies)
            for freq in freqs:
                # Find the closest frequency bin to the target frequency
                bin_idx = np.argmin(np.abs(frequencies - freq))
                #for i in range(bin_idx-5,bin_idx+5): allowed take more bins, but actally makes only worse
                #    mask[i] = 1  # Set the corresponding frequency bin to 1
                mask[bin_idx]=1    
            masks[key] = mask


        # Step 5: Detect keys based on masks and time slices
        #detected_keys = []  # List to store detected keys
        masked_sums =[] # list to store dot products
        # Iterate through each time slice
        for t in range(stft_filtered.shape[1]):
            time_slice = np.abs(stft_filtered[:, t])  # Get the magnitude of the t-th time slice
            max_sum=0
            for key, mask in masks.items():
                # Perform scalar multiplication (dot product) of mask with time slice
                masked_sum = np.dot(mask, time_slice)
                if masked_sum > max_sum:
                    max_sum = masked_sum

            if masked_sum != 0:
                masked_sums.append(masked_sum)

        print(masked_sums)
        new_detected_keys = []
        for t in range(stft_filtered.shape[1]):
            time_slice = np.abs(stft_filtered[:, t])  # Get the magnitude of the t-th time slice
            max_key="-"
            max_sum=0
            for key, mask in masks.items():
                # Perform scalar multiplication (dot product) of mask with time slice
                masked_sum = np.dot(mask, time_slice)
                #print(key, masked_sum)
                if masked_sum > max_sum:
                    max_key = key
                    max_sum = masked_sum
                # If the sum is not zero, the key's frequencies exist
            if max_sum > np.median(masked_sums)*SETTINGS['MASKS_MEDIAN_MULTIPLIER']  or max_key=="-":
                new_detected_keys.append(max_key)

        print(new_detected_keys)
        final_keys=list()
        is_another = False
        final_keys.append(new_detected_keys[0])
        for i in new_detected_keys:
            if i=="-":
                is_another = True
                continue
            elif final_keys[-1]!=i or is_another :
                final_keys.append(i)
                is_another = False

        print(final_keys)



























        
        bot.reply_to(message, f"Masks median is {str(np.median(masked_sums))}")
        #apply masks to get keys and check for median
        print(new_detected_keys)
        bot.reply_to(message, f"Here are preliminary detected keys {','.join(new_detected_keys)}")
        print(final_keys)
        bot.reply_to(message, f"Here are final keys: {','.join(final_keys)}")
        
        

        


    
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {str(e)}")

# Handler for settings commands
@bot.message_handler(commands=['set'])
def handle_set_command(message):
    try:
        # Parse the command input
        command_parts = message.text.split()
        if len(command_parts) != 3:
            bot.reply_to(message, "Usage: /set <setting_name> <value>")
            return
        
        setting_name, value = command_parts[1], command_parts[2]
        
        # Validate the setting
        if setting_name not in SETTINGS:
            bot.reply_to(message, f"Unknown setting: {setting_name}. Available settings: {', '.join(SETTINGS.keys())}")
            return
        
        # Cast the value to the appropriate type
        if isinstance(SETTINGS[setting_name], int):
            value = int(value)
        elif isinstance(SETTINGS[setting_name], float):
            value = float(value)
        else:
            bot.reply_to(message, f"Invalid type for setting: {setting_name}")
            return
        
        # Update the setting
        SETTINGS[setting_name] = value
        bot.reply_to(message, f"Setting updated: {setting_name} = {value}")
    
    except ValueError:
        bot.reply_to(message, f"Invalid value type for setting: {setting_name}. Please provide a valid number.")
    except Exception as e:
        bot.reply_to(message, f"An error occurred: {str(e)}")

@bot.message_handler(commands=['getsettings'])
def handle_get_command(message):
    text="Setting name\t:\tvalue\n"
    for setting_name,value in SETTINGS.items():
        text+=f'{setting_name}\t:\t{value}\n'
    bot.reply_to(message, text)

    
# Handle all other messages with content_type 'text' (content_types defaults to ['text'])
@bot.message_handler(content_types=['text'])
def echo_message(message):
    bot.reply_to(message, "~"+message.text+"~",parse_mode="MarkdownV2")

bot.infinity_polling()