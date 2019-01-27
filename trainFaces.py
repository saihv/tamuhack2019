import cognitive_face as CF

key = 'db9b516ba0414434a511351345550f91' # make sure to fill in the key you obtained for Face API
BASE_URL = 'https://southcentralus.api.cognitive.microsoft.com/face/v1.0'  # Replace with your regional Base URL
PERSON_GROUP_ID = 'auth-person'

def initRemoteInfo():
    CF.Key.set(key)
    CF.BaseUrl.set(BASE_URL)

def createGroup():
    CF.person_group.create(PERSON_GROUP_ID, 'Known Persons')

def createPerson(name):
    user_data = 'More information can go here'
    response = CF.person.create(PERSON_GROUP_ID, name, user_data)
    person_id = response['personId']
    return person_id

def addFace(person_id, filename):
    CF.person.add_face(filename, PERSON_GROUP_ID, person_id)

def trainModel():
    CF.person_group.train(PERSON_GROUP_ID)

def checkTrainingStatus():
    response = CF.person_group.get_status(PERSON_GROUP_ID)
    status = response['status']
    print(status)
    return status

if __name__ == '__main__':
    initRemoteInfo()

    #createGroup()
    name = 'George Chustz'
    pId = createPerson(name)

    numTrainImages = 5

    for i in range(numTrainImages):
        filename = 'george' + str(i+1) + '.png'
        addFace(pId, filename)

    trainModel()
    while True:
        trainStatus = checkTrainingStatus()
        if str(trainStatus) == 'succeeded':
            break
