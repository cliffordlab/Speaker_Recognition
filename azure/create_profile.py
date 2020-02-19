import http.client, urllib.request, urllib.parse, urllib.error, base64

headers = {
    # Request headers
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': '9bb3ecfbe03f4c039c703204d8d64c0e',
}
url = 'https://spkr-trial.cognitiveservices.azure.com/spid/v1.0/identificationProfiles'
params = urllib.parse.urlencode({
})
body={"locale":"en-us"}

try:
    conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
    print(params, body, headers)
    conn.request("POST", "/spid/v1.0/identificationProfiles?%s" % params, body, headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))

####################################
