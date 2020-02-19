########### Python 3.2 #############
import http.client, urllib.request, urllib.parse, urllib.error, base64
import soundfile as sf
import wave

headers = {
    # Request headers
    'Content-Type': 'multipart/form-data',
    'Ocp-Apim-Subscription-Key': '9bb3ecfbe03f4c039c703204d8d64c0e',
}

params = urllib.parse.urlencode({
    # Request parameters
    'shortAudio': 'true',
})
wavfile = '/Users/sanmathikamath/projects/datasets/arctic/bdl/arctic_a0578.wav'
w = wave.open(wavfile, 'rb')
binary_data = w.readframes(w.getnframes())
w.close()
with open(wavfile ,'rb') as f:
    x=f.read()
#body, sr = sf.read(wavfile)
body = x
try:
    conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
    conn.request("POST", "/spid/v1.0/identificationProfiles/fcedd3a9-4a03-42e0-bc0d-adec05ce675d/enroll?%s" % params, "f", headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))

####################################
