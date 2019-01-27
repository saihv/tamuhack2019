import pyaudio
import wave
import SpeechModules.VerificationServiceHttpClientHelper as VerificationService
import sys

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 4

numTrainExamples = 10

APIKey = 'a7372791257749e28ac394767ae5dca7'
# profile = '2f439ad3-c9c9-4269-8203-fbbed76f9719'
profile = 'bda3ad98-59da-4e4c-94c8-402129ef8820'

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

def enrollSample(agent, filename):
    status = 0
    try:
        enrollment_response = agent.enroll_profile(profile, filename)
        print('Enrollments Completed = {0}'.format(enrollment_response.get_enrollments_count()))
        print('Remaining Enrollments = {0}'.format(enrollment_response.get_remaining_enrollments()))
        print('Enrollment Status = {0}'.format(enrollment_response.get_enrollment_status()))
        print('Enrollment Phrase = {0}'.format(enrollment_response.get_enrollment_phrase()))

        status = 1
    except Exception:
        pass

    return status

if __name__ == '__main__':
    verificationAgent = VerificationService.VerificationServiceHttpClientHelper(APIKey)
    idx = 0
    while True:
        filename = 'trainSample_' + str(idx) + '.wav'
        recordStream(filename)
        enrollStatus = enrollSample(verificationAgent, filename)

        if enrollStatus == True:
            idx += 1
        if idx == numTrainExamples:
            break
            

    
