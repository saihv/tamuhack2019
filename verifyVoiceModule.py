import pyaudio
import wave
import SpeechModules.VerificationServiceHttpClientHelper as VerificationService
import sys

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 4

apiKey = "a7372791257749e28ac394767ae5dca7"
#profileID = "1d998ccb-3b5f-4386-a71f-2c90e81336fb"
profileID = 'bda3ad98-59da-4e4c-94c8-402129ef8820'

def recordStream(wavFilename):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    print("Recording sample...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Done.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(wavFilename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def verifySample(agent, subscription_key, file_path, profile_id):
    """verify a profile based on submitted audio sample

    Arguments:
    subscription_key -- the subscription key string
    file_path -- the audio file path for verification
    profile_id -- ID of a profile to attempt to match the audio sample to
    """
    authorized = False
    verification_response = agent.verify_file(file_path, profile_id)
    print('Verification Result = {0}'.format(verification_response.get_result()))
    print('Confidence = {0}'.format(verification_response.get_confidence()))

    if str(verification_response.get_result()) == 'Accept':
        authorized = True

    return authorized, str(verification_response.get_confidence())

def verifyAudio():
    verificationAgent = VerificationService.VerificationServiceHttpClientHelper(
        apiKey)
    
    filename = 'TestRecord.wav'
    recordStream(filename)
    authStatus, conf = verifySample(verificationAgent, apiKey, filename, profileID)

    return authStatus, conf