import csv
import json
from collections import defaultdict
from copy import copy, deepcopy
from obspy import UTCDateTime as UTC, read_inventory
from obspy.clients.nrl import NRL
from obspy.core.inventory import Inventory, Network, Station, Channel, Site, Equipment
from obspy.io.sh.core import from_utcdatetime
#from obspy.core.util import AttribDict
import os.path



def csv2list(name):
    fname  = TNAME.format(name)
    with open(fname) as f:
        reader = csv.reader(f)
        return list(reader)[1:]

def csv2dictlist(name):
    fname  = TNAME.format(name)
    d = defaultdict(list)
    with open(fname) as f:
        reader = csv.DictReader(f)
        for line in reader:
            if line['station code'] != '':
                key = line['station code']
                d[key].append(line)
    return d

def _strip_keys(keys):
    return [k.strip() for k in keys.split('|')]

def load_gain():
    return {line[0]: line[2:5] if line[2] != '' else None
                for line in csv2list('seismometer')}

def load_seism_NRL():
    lines = csv2list('seismometer_NRL')
    return {line[0]: _strip_keys(line[1]) for line in lines}

def load_digi_NRL():
    lines = csv2list('digitizer_NRL')
    return {line[0]: _strip_keys(line[1]) for line in lines}

def _date2sh(utc):
    return '...' if utc is None else from_utcdatetime(utc).lower()[:17]


def write_gain_expressions_for_table():
    """Write expressions for google tables into file"""
    seism_NRL = load_seism_NRL()
    digi_NRL = load_digi_NRL()
    nrl = NRL()
    for v in digi_NRL.values():
        v[-1] = v[-1].format(sr=100)
    # get sensitivities
    digi_sens = {k: nrl.get_datalogger_response(v).instrument_sensitivity.value for k, v in digi_NRL.items()}
    seism_sens = {k: nrl.get_sensor_response(v).instrument_sensitivity.value for k, v in seism_NRL.items()}
    expr = ('expression for digi_NRL:\n=SWITCH(A2, {}, "")\n\n'
            'expression for seism_NRL:\n=SWITCH(A2, {}, "")')
    ins1 = ', '.join('"{}", {}'.format(k, v) for k, v in digi_sens.items())
    ins2 = ', '.join('"{}", {}'.format(k, v) for k, v in seism_sens.items())
    with open(OUT + 'expressions_for_table.txt', 'w') as f:
        f.write(expr.format(ins1, ins2))


def setdefault(dict_, key, default=None):
    """Similar to dict.setdefault"""
    if dict_[key] == '':
        dict_[key] = default
    return dict_[key]

class ConfigJSONDecoder(json.JSONDecoder):
    """Decode JSON config with comments stripped"""
    def decode(self, s):
        s = '\n'.join(l.split('#', 1)[0] for l in s.split('\n'))
        return super(ConfigJSONDecoder, self).decode(s)

def meta2xml(only_public=False):
    """Write StationXML, SH sensitivity and statinf"""
    stations = []
    shm_sens = []
    shm_loc = []
    info = []
    # tsn is dictionary {station_code: epoch_list}
    tsn = csv2dictlist(NET_CODE)
    # sort epochs by time, coordinates of latest epoch will be used as station
    # coordinates
    for sta in tsn:
        tsn[sta] = sorted(tsn[sta], key = lambda epoch: epoch['UTC_starttime'])
    seism_NRL = load_seism_NRL()
    digi_NRL = load_digi_NRL()
    gain = load_gain()
    nrl = NRL()
    for sta in tsn:
        sta_startdate = None
        channels = []
        for epoch in tsn[sta]:
            if only_public and epoch['public'] != 'TRUE':
                continue
            setdefault(epoch, 'seismometer id')
            try:
                gain_values = gain[epoch['seismometer id']]
            except KeyError:
                gain_values = None
            sta_code = epoch['station code']
            loc_cha_code = setdefault(epoch, 'location and channel (non default)', DEFAULT_LOC_CHA_CODE)
            loc_code, cha_code_template = loc_cha_code.split('.')
            srs = setdefault(epoch, 'sampling rates', DEFAULT_SRS)
            srs = [int(sr.strip()) for sr in srs.split(',')]


            startdate = UTC(epoch['UTC_starttime'])
            enddate = UTC(epoch['UTC_endtime']) if epoch['UTC_endtime'] != '' else None
            if sta_startdate is None:
                sta_startdate = startdate
            try:
                lat = float(epoch['latitude'])
                lon = float(epoch['longitude'])
                elev = float(epoch['elevation'])
                k1 = digi_NRL[epoch['digitizer']]
                k2 = seism_NRL[epoch['seismometer type']]
            except (ValueError, KeyError):
                lat, lon, elev, k1, k2 = None, None, None, None, None
                msg = ('Ignore epoch {station code} {UTC_starttime} -- '
                       'problem with coordinates, seismometer or digitizer')
                print(msg.format(**epoch))
                continue
            data_logger = Equipment(manufacturer=k1[0],
                                    description=epoch['digitizer'],
                                    model=epoch['digitizer'].split('_')[0])
            sensor = Equipment(manufacturer=k2[0],
                               description=epoch['seismometer type'],
                               model=epoch['seismometer type'],
                               serial_number=epoch['seismometer id'])

            for sr in srs:
                k1_sr = copy(k1)
                k1_sr[-1] = k1_sr[-1].format(sr=sr)
                response = nrl.get_response(k1_sr, k2)
                sum_sens = 0
                if k2 == ['HGS Products','HG-6','4.5 Hz','9090 Ohms (B coil)']:
                    reffreq = response.instrument_sensitivity.frequency
                else:
                    reffreq = 0.4 if sr < 2 else 1.0
                # see obspy PR #2248
                stage0 = response.response_stages[0]
                response.instrument_sensitivity.input_units = stage0.input_units
                response.instrument_sensitivity.input_units_description = stage0.input_units_description
                response.recalculate_overall_sensitivity(reffreq)
                if epoch['intern remarks'] == '!NRLdigi':
                    digita = {line[0]: _strip_keys(line[2]) for line in csv2list('digitizer_NRL')}
                    response.response_stages[2].stage_gain = float(digita[data_logger.description][0])
                for comp in 'ZNE':
                    cha_code = cha_code_template.replace('?', SR2CODE[sr]) + comp
                    seed_id = '.'.join([NET_CODE, sta_code, loc_code, cha_code])
                    path1 = f'{RESP}{seed_id}_{startdate!s:.10}.resp'
                    path2 = f'{RESP}{seed_id}.resp'
                    path = path1 if os.path.exists(path1) else path2
                    if os.path.exists(path):
                        print(f'  Found and use response file at {path}')
                        chresponse = read_inventory(path)[0][0][0].response
                        chresponse.recalculate_overall_sensitivity(reffreq)
                    else:
                        # adjust gain in response
                        if gain_values is not None:
                            chresponse = deepcopy(response)
                            chresponse.response_stages[0].stage_gain = float(gain_values['ZNE'.index(comp)])
                            chresponse.recalculate_overall_sensitivity(reffreq)
                        else:
                            chresponse = response
                    sum_sens += chresponse.instrument_sensitivity.value
                    # create channel
                    kw = dict(code=cha_code, location_code=loc_code,
                              latitude=lat, longitude=lon,
                              elevation=elev, depth=0,
                              azimuth=COMP2AZIDIP[comp][0],
                              dip=COMP2AZIDIP[comp][1],
                              sample_rate=sr, response=chresponse,
                              start_date=startdate, end_date=enddate,
                              storage_format=STORAGE,
                              sensor=sensor, data_logger=data_logger)
                    channels.append(Channel(**kw))
                    fargs = (sta_code.lower(), cha_code.lower()[:2], comp.lower(),
                             _date2sh(startdate), _date2sh(enddate),
                             1e9 / chresponse.instrument_sensitivity.value)
                    if epoch['seismometer type'] != 'CMG-5T':  # ignore accelerometers
                        shm_sens.append(SHM_SENS.format(*fargs))
                # check that sensitivity values in table are correct within 1%
                sens = sum_sens / 3
                sens_table = float(epoch['count/m*s(*s)'])
                error = abs(sens_table - sens) / sens
                if error > 0.01:
                    print('epoch {station code} {UTC_starttime} -- '.format(**epoch) +
                          'error in sensitivity reported in TSN table, loc.cha {}.{}, expected: {:.3e}, got: {:.3e}, difference: {:.2f}%'.format(loc_code, cha_code, sens, sens_table, error * 100))
            if epoch['auxStream']:
                auxStreams = epoch['auxStream'].split(',')
                for aux in auxStreams:
                    kw = dict(code=aux, location_code=loc_code,
                              latitude=lat, longitude=lon,
                              start_date=startdate, end_date=enddate,
                              elevation=elev, depth=-9999,)
                    channels.append(Channel(**kw))
        if len(channels) == 0:
            continue
        name = epoch['name']
        fargs = (sta_code.upper(), lat, lon, elev, name, 1e9 / sens)
        shm_loc.append(SHM_LOC.format(*fargs))
        fargs = (sta_code, name, lat, lon, elev,
                 sens, 1e9 / sens,
                 chresponse.get_paz().normalization_factor,
                 chresponse.get_paz().zeros, chresponse.get_paz().poles)
        info.append(INFO.format(*fargs))
        stations.append(Station(code=sta_code, latitude=lat, longitude=lon,
                      elevation=elev, creation_date=sta_startdate,
                      restricted_status='open',
                      site=Site(name=STA_NAME.format(name), country='Germany'),
                      channels=channels,
                      start_date=sta_startdate, end_date=enddate))
    net = Network(code=NET_CODE, stations=stations, description=NET_DESC,
                  start_date=UTC(NET_START), restricted_status='open')
    inv = Inventory(networks=[net], source=SOURCE)
    fname = NET_CODE + '_private' * (not only_public)
    inv.write(OUT + fname + '.xml', 'STATIONXML', validate=True)
    with open(OUT + 'SH_' + fname + '_statinf.dat', 'w') as f:
        f.write('\n'.join(shm_loc))
    with open(OUT + 'SH_' + fname + '_sensitivities.txt', 'w') as f:
        f.write('\n'.join(shm_sens))
    print()
    print('PUBLIC: ', only_public)
    print(INFO_INFO)
    print('\n'.join(info))
    print()





with open('config.json') as f:
    conf = json.load(f, cls=ConfigJSONDecoder)
globals().update(conf)   

STORAGE = 'Steim2'

TNAME = 'Stations_-_{}.csv'
DEFAULT_LOC_CHA_CODE = '.?H'
DEFAULT_SRS = '100'
SR2CODE = {200: 'H', 100: 'H', 20: 'B', 1: 'L'}
COMP2AZIDIP = {'Z': (0, -90), 'N': (0, 0), 'E': (90, 0)}

SHM_LOC = '{:5}  lat:{:+.6f}  lon:{:+.6f}  elevation:{:.1f}  name:{}'
SHM_SENS = '{}-{}-{} {} {} {:.5f}'
INFO_INFO = 'code  name                      lat     lon     elev   gain      SH    PAZ'
INFO = '{:5} {:25} {:.3f}  {:.3f}  {:.1f}  {:.2e}  {:.2f}  norm:{:.2e} zeros:{} poles:{}'


write_gain_expressions_for_table()
meta2xml(only_public=False)
meta2xml(only_public=True)
