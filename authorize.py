import verifyImageModule
import verifyVoiceModule

if __name__ == '__main__':
    imageAuthStatus, conf = verifyImageModule.verifyImage()
    if imageAuthStatus:
        conf = conf*100
        if conf > 50 and conf < 70:
            confString = 'Medium'
        elif conf > 70:
            confString = 'High'
        else:
            confString = 'Low'
        print('Image authenticated with ' + confString + ' confidence.')
    speechAuthStatus, conf = verifyVoiceModule.verifyAudio()
    if speechAuthStatus:
        print('Speech authenticated with ' + conf + ' confidence.')
    else:
        print('Speech authentication failed.')

    if imageAuthStatus and speechAuthStatus:
        print("Authentication successful! Access granted.")
    else:
        print("Authentication failed!")
    
