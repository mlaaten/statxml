import requests
url = "https://docs.google.com/spreadsheets/d/{key}/gviz/tq?tqx=out:csv&sheet={sheet}"
with open('KEY') as f:
    key = f.read()
sheets = ('TH', 'SX', 'seismometer', 'seismometer_NRL', 'digitizer', 'filters')
out = 'metadata/Stations_{sheet}.csv'
for sheet in sheets:
    response = requests.get(url.format(key=key, sheet=sheet))
    assert response.status_code == 200, 'Wrong status code'
    with open(out.format(sheet=sheet), 'wb') as f:
        f.write(response.content)
