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
    'MAGNITUDE_AMP_MULTIPLIER':25.0, #by how much increase the amplituted of DTMF freqs before applying threshold filtering
    'THRESHOLD_PERCENTILE': 99.0, #which percentile to use as a threshold
    'MASKS_MEDIAN_MULTIPLIER':10.0, #by how much to multiply median when extractin keys
    'PLT_YLIM_LOWER':0, #lower freq to show
    'PLT_YLIM_UPPER':16384, #upper freq to show
    'FIG_SIZE_X':10, # control width of diagram
    'FIG_SIZE_Y':4, # control height of diagram
    #'PLT_XLIM_LOWER':0,
   # 'PLT_XLIM_UPPER':2048,


}


bot = telebot.TeleBot(API_TOKEN)
def generate_spectrogram_image(spectrogram,sr,hop_length):
    # Read audio from the buffer with soundfile
    print("Plotting")
    plt.figure(figsize=(SETTINGS['FIG_SIZE_X'], SETTINGS['FIG_SIZE_Y']))    librosa.display.specshow(spectrogram,hop_length=hop_length, sr=sr, x_axis='time', y_axis='log')
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
    image_buffer.seek(0)
    return image_buffer

def fetch_audio_as_buffer(file_id):
    file_info = bot.get_file(file_id)
    file_url = f"https://api.telegram.org/file/bot{API_TOKEN}/{file_info.file_path}"
    response = requests.get(file_url)
    response.raise_for_status()
    return io.BytesIO(response.content)

def threshold_filtering(magnitude,phase,freq_min,freq_max,sr,n_fft):
    #generate freq from bins
    print("thresholding")
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Step 3: Identify frequency bins within the desired range
    freq_range = (frequencies >= freq_min) & (frequencies <= freq_max)  # Boolean mask for frequency range
    magnitude_in_range = magnitude[freq_range, :]  # Extract magnitudes in this range

    # Calculate the maximum magnitude in this range
    #max_magnitude = np.max(magnitude)
    #calculate median
    #median_magnitude = np.median(magnitude)

    #Compute the threshold as half of the median
    #threshold = median_magnitude *15
    # Compute the threshold as half of the maximum
    #threshold = max_magnitude /5
    #compute threshold depending on percentile
    n = SETTINGS['THRESHOLD_PERCENTILE'] # Define the percentile
    percentile_magnitude = np.percentile(magnitude_in_range, n)
    threshold = percentile_magnitude
    #threshold =25.059767
    dtmf_frequencies = [697, 770, 852, 941, 1209, 1336, 1477, 1633]

# Step 2: Get the frequency bins corresponding to DTMF tones
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    dtmf_bins = [np.argmin(np.abs(frequencies - freq)) for freq in dtmf_frequencies]

# Step 3: Amplify the DTMF frequencies by 10 times
    amplified_magnitude = np.copy(magnitude)
    for bin_idx in dtmf_bins:
        #for i in range(max(0, bin_idx - 2), min(len(frequencies), bin_idx + 3)):
        amplified_magnitude[bin_idx, :] *= SETTINGS['MAGNITUDE_AMP_MULTIPLIER']  


    print(threshold)
    stft_filtered = amplified_magnitude * (np.abs(amplified_magnitude) > threshold) * phase
    return stft_filtered

#def compute_median_dtmf_masks(stft_result,sr,n_fft):
#    # Get the frequencies corresponding to STFT rows
#    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
#    DTMF_FREQUENCIES = {
#    '1': [697, 1209],
#    '2': [697, 1336],
#    '3': [697, 1477],
#    '4': [770, 1209],
#    '5': [770, 1336],
#    '6': [770, 1477],
#    '7': [852, 1209],
#    '8': [852, 1336],
#    '9': [852, 1477],
#    '0': [941, 1336],
#    '*': [941, 1209],
#    '#': [941, 1477],
#    'A':[697,1633],
#    'B':[770,1633],
#    'C':[852,1633],
#    'D':[941,1633]}
#    # Step 4: Generate masks for each key
#    masks = {}
#    for key, freqs in DTMF_FREQUENCIES.items():
#        mask = np.zeros_like(frequencies)
#        print(freqs)
#        for freq in freqs:
#            # Find the closest frequency bin to the target frequency
#            bin_idx = np.argmin(np.abs(frequencies - freq))
#            #for i in range(bin_idx-500,bin_idx+500):
#            #    mask[i] = 1  # Set the corresponding frequency bin to 1
#            mask[bin_idx]=1    
#        masks[key] = mask
#        print(mask)
#    masked_sums =[] #list to store a sum after masks applied
#    # Iterate through each time slice (column in stft_result)
#    for t in range(stft_result.shape[1]):
#        time_slice = np.abs(stft_result[:, t])  # Get the magnitude of the t-th time slice
#        max_sum=0.0
#        for _, mask in masks.items():
#            # Perform scalar multiplication (dot product) of mask with time slice
#            masked_sum = np.dot(mask, time_slice)
#            print(masked_sum)
#            if masked_sum > max_sum:
#                max_sum = masked_sum
#            # If the sum is not zero, the key's frequencies exist
#        if masked_sum != 0:
#            masked_sums.append(masked_sum)
#        print(masked_sums)
#        return np.mean(masked_sums)
def compute_median_dtmf_masks(stft_result,sr,n_fft):
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
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    tolerance_bins=5
    # Step 4: Generate masks for each key
    masks = {}
    for key, freqs in dtmf_frequencies.items():
        mask = np.zeros_like(frequencies)
        print(freqs)
        for freq in freqs:
            # Find the closest frequency bin to the target frequency
            bin_idx = np.argmin(np.abs(frequencies - freq))
            #for i in range(max(0, bin_idx - tolerance_bins), min(len(mask), bin_idx + tolerance_bins)):
            #    mask[i] = 1  # Set the corresponding frequency bin to 1
            mask[bin_idx]=1    
        masks[key] = mask
        print(mask)

    # Step 5: Detect keys based on masks and time slices
    detected_keys = []  # List to store detected keys
    masked_sums =[]
    # Iterate through each time slice (column in D)
    for t in range(stft_result.shape[1]):
        time_slice = np.abs(stft_result[:, t])  # Get the magnitude of the t-th time slice
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
        detected_keys.append(max_key)
        if masked_sum != 0:
            masked_sums.append(masked_sum)

            #if no key detected, add -1
            #detected_keys.append("-1")

    # Step 6: Display the results
    print("Detected DTMF keys:", detected_keys)
    print(masked_sums)
    return np.median(masked_sums)


def detect_keys_preliminary(stft_result,sr,n_fft,masked_sums_median):
    preliminary_detected_keys = []
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    DTMF_FREQUENCIES = {
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
    masks ={}
    for key, freqs in DTMF_FREQUENCIES.items():
        mask = np.zeros_like(frequencies)
        for freq in freqs:
            # Find the closest frequency bin to the target frequency
            bin_idx = np.argmin(np.abs(frequencies - freq))
            #for i in range(bin_idx-50,bin_idx+50):
            #for i in range(max(0, bin_idx - 5), min(len(mask), bin_idx + 5)):
            #    mask[i] = 1  # Set the corresponding frequency bin to 1
            mask[bin_idx]=1
        print(np.sum(mask))    
        masks[key] = mask
    for t in range(stft_result.shape[1]):
        time_slice = np.abs(stft_result[:, t])  # Get the magnitude of the t-th time slice
        max_key="-"
        max_sum=0
        for key, mask in masks.items():
            # Perform scalar multiplication (dot product) of mask with time slice
            masked_sum = np.dot(mask, time_slice)
            print(masked_sum)
            if masked_sum > max_sum:
                max_key = key
                max_sum = masked_sum
            # If the sum is not zero, the key's frequencies exist
        if max_sum > masked_sums_median*SETTINGS['MASKS_MEDIAN_MULTIPLIER']  or max_key=="-":
            preliminary_detected_keys.append(max_key)
        else:
            preliminary_detected_keys.append("-")
        print(preliminary_detected_keys)
        return preliminary_detected_keys

def extract_final_keys(prelim_keys):
    final_keys=list()
    is_another = False
    final_keys.append(prelim_keys[0])
    for i in prelim_keys:
        if i=="-":
            is_another = True
            continue
        elif final_keys[-1]!=i or is_another :
            final_keys.append(i)
            is_another = False

    return final_keys
# Handle '/start' and '/help'
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, """\
Hello Agent! I am your tool to analyse DTMF frequencies!\
""")

@bot.message_handler(commands=['help'])
def send_welcome(message):
    bot.reply_to(message, """\
Just record me an audio message!\
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

            # Check the file type using magic bytes
            #mime = magic.Magic(mime=True)
            #file_type = mime.from_buffer(audio_buffer.getvalue())
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
        #audio_buffer = fetch_audio_as_buffer(file_id)
        audio_buffer.seek(0)  # Ensure the buffer is at the beginning
        #y, sr = sf.read(audio_buffer, dtype='float32')
        y, sr = librosa.load(audio_buffer,sr=None)

        print(sr)
    # If audio is stereo, convert it to mono
        if len(y.shape) > 1:
            y = librosa.to_mono(y.T)
        #n_fft=8820//2
        #win_length = 8820//2
        #hop_length = 8820//4
    # Compute the Short-Time Fourier Transform (STFT)
        stft_result = librosa.stft(y,n_fft=SETTINGS['N_FFT'],  win_length=SETTINGS['WIN_LEN'],hop_length=SETTINGS['HOP_LEN'], window='hann', center=True, dtype=None, pad_mode='constant', out=None)
        magnitude,phase = librosa.magphase(stft_result)
        spectrogram = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
    
        # Generate spectrogram image
        spectrogram_image = generate_spectrogram_image(spectrogram,sr,SETTINGS['HOP_LEN'])
        

        # Send the spectrogram image back to the user
        bot.send_photo(message.chat.id, spectrogram_image, caption="Here is the spectrogram of initial audio")
        # threshold filtering
        stft_result_threshold_filtered = threshold_filtering(magnitude,phase,691,1640,sr,SETTINGS['N_FFT'])

        spectrogram_filtered  = librosa.amplitude_to_db(np.abs(stft_result_threshold_filtered), ref=np.max)
        spectrogram_image_filtered = generate_spectrogram_image(spectrogram_filtered,sr,SETTINGS['HOP_LEN'])

        bot.send_photo(message.chat.id, spectrogram_image_filtered, caption="Here is the spectrogram of cleaned audio")
        bot.reply_to(message, "Calculating the masks median",parse_mode="MarkdownV2")
        masks_sum_median = compute_median_dtmf_masks(stft_result_threshold_filtered,sr,SETTINGS['N_FFT'])
        bot.reply_to(message, f"Masks median is {str(masks_sum_median)}")
        #apply masks to get keys and check for median
        prelim_keys = detect_keys_preliminary(stft_result_threshold_filtered,sr,SETTINGS['N_FFT'],masks_sum_median)
        print(prelim_keys)
        bot.reply_to(message, f"Here are preliminary detected keys {','.join(prelim_keys)}")
        final_keys = extract_final_keys(prelim_keys)
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