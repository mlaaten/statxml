# MIT license, Copyright Tom Eulenfeld
"""
Changes
dev
  * specify different config file with `-c conf_SX.json` and other new cli arguments
  * set default config file name to conf.json
  * support using offline NRL (faster) by setting NRL config parameter
  * support network DOI
  * specify seismometer responses with poles and zeros or corner frequency and damping
  * subtract new depth column from elevation for channel epochs (StationXML definition)
  * major refactoring
v0.1.0
  * start versioning

"""
import argparse
import csv
import json
from collections import defaultdict
from copy import copy, deepcopy
import numpy as np
from obspy import UTCDateTime as UTC
from obspy.clients.nrl import NRL as NRLClient
from obspy.core.inventory import Inventory, Network, Station, Channel, Site, Equipment
from obspy.core.inventory.response import CoefficientsTypeResponseStage, PolesZerosResponseStage, Response, InstrumentSensitivity
from obspy.io.sh.core import from_utcdatetime
from obspy.signal.invsim import corn_freq_2_paz

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iter_):
        return iter_
else:
    # hack, see https://stackoverflow.com/a/38345993
    old_print = print
    def new_print(*args, **kwargs):
        try:
            tqdm.write(*args, **kwargs)
        except:
            old_print(*args, ** kwargs)
    print = new_print


class ConfigJSONDecoder(json.JSONDecoder):
    """Decode JSON config with comments stripped"""
    def decode(self, s):
        s = '\n'.join(l.split('#', 1)[0] for l in s.split('\n'))
        return super(ConfigJSONDecoder, self).decode(s)

def csv2list(name):
    fname  = CONF['TNAME'].format(name)
    with open(fname) as f:
        reader = csv.reader(f)
        return list(reader)[1:]

def csv2dictlist(name, colkey='station code'):
    fname  = CONF['TNAME'].format(name)
    d = defaultdict(list)
    with open(fname) as f:
        reader = csv.DictReader(f)
        for line in reader:
            if line[colkey] != '':
                key = line[colkey]
                d[key].append(line)
    return d

def csv2dict(name, colkey='name'):
    fname  = CONF['TNAME'].format(name)
    d = {}
    with open(fname) as f:
        reader = csv.DictReader(f)
        for line in reader:
            if line[colkey] != '':
                d[line[colkey]] = line
    return d

def _strip_keys(keys):
    return [k.strip() for k in keys.split('|')]

def _setdefault(dict_, key, default=None):
    """Similar to dict.setdefault"""
    if dict_[key] == '':
        dict_[key] = default
    return dict_[key]

def load_tables():
    # seismometer -> dict of NRL keys, PAZ, etc
    seismometers = csv2dict('seismometer', colkey='seismometer')
    for line in seismometers.values():
        for k in line:
            _setdefault(line, k)  # replace '' with None
        for k in ('NRLv1', 'NRLv2'):
            if line.get(k):
                line[k] = _strip_keys(line[k])
        for k in ('poles', 'zeros'):
            if line[k]:
                line[k] = list(map(complex, line[k].split(',')))
        for k in ('sensitivity V/m*s(*s)', 'corner period (s)', 'damping', 'reffreq (Hz)'):
            if line[k]:
                line[k] = float(line[k])
    # seismometer id -> gain Z, N, E (different from NRL)
    seismometers_custom_gain_comp = {
        line[0]: list(map(float, line[2:5]))
        for line in csv2list('seismometer_id') if line[2] != ''}
    # digitizer -> dict of NRL keys, custom sensitivity
    digitizers = csv2dict('digitizer', colkey='digitizer')
    for line in digitizers.values():
        for k in line:
            _setdefault(line, k)  # replace '' with None
        for k in ('NRLv1', 'NRLv2'):
            if line.get(k):
                line[k] = _strip_keys(line[k])
        k = 'sensitivity count/V'
        if line[k]:
            line[k] = float(line[k])
    # filter name -> coefficients
    decimation_filters = csv2dict('filter')
    for line in decimation_filters.values():
        line['coefficients'] = list(map(float, line['coefficients'].split()))
        for k in ('decimation', 'number'):
            line[k] = int(line[k])
        line['symmetry'] = line['symmetry'].upper()
    return (
        seismometers,
        seismometers_custom_gain_comp,
        digitizers,
        decimation_filters
        )


def _date2sh(utc):
    return '...' if utc is None else from_utcdatetime(utc).lower()[:17]


def calc_fc(response):
    f = np.logspace(-3, 2, 1000)
    r = np.abs(response.get_evalresp_response_for_frequencies(f))
    i = np.nonzero(r>=np.max(r)/2**0.5)[0][0]
    return f[i]

def calc_norm(fc, damp, reffreq=1):
    s = 2j*np.pi*reffreq
    resp_func = s**2/(s**2 + 2*damp*fc*s + fc**2)
    return 1 / np.abs(resp_func)


def write_info(print_=False):
    """Write out expressions and PAZ info"""
    digitizers = deepcopy(DIGITIZERS)
    for v in digitizers.values():
        if v[NRLKEY]:
            v[NRLKEY][-1] = v[NRLKEY][-1].format(sr=100)

    # get sensitivities
    digi_sens = {k: NRL.get_datalogger_response(v[NRLKEY]).instrument_sensitivity.value for k, v in digitizers.items() if v[NRLKEY]}
    seism_sens = {k: NRL.get_sensor_response(v[NRLKEY]).instrument_sensitivity.value for k, v in SEISMOMETERS.items() if v[NRLKEY]}
    expr = ('Expression for digiitizer tab:\n=SWITCH(A2, {}, "")\n\n'
            'Expression for seismometer tab:\n=SWITCH(A2, {}, "")\n\n')
    ins1 = ', '.join('"{}", {}'.format(k, v) for k, v in digi_sens.items())
    ins2 = ', '.join('"{}", {}'.format(k, v) for k, v in seism_sens.items())
    out = expr.format(ins1, ins2)

    out = out + 'PAZ for corner period + damping\n'
    for seism, line in SEISMOMETERS.items():
        if line['corner period (s)']:
            fc = 1 / float(line['corner period (s)'])
            damp = float(line['damping'])
            resp = corn_freq_2_paz(fc, damp)
            reffreq = line['reffreq (Hz)'] or 1
            norm = calc_norm(fc, damp, reffreq=reffreq)
            poles = resp['poles']
            expr = f'{seism} -- poles: {poles[0]:.5f}, {poles[1]:.5f}, normalization: {norm:.3f} @ {reffreq}Hz\n'
            out = out + expr
    out = out + '\nPAZ from NRL\n'
    for seism, v in  SEISMOMETERS.items():
        if v[NRLKEY]:
            resp = NRL.get_sensor_response(v[NRLKEY])
            st = resp.response_stages[0]
            expr = f'{seism}\n  poles: {st.poles}\n  zeros: {st.zeros}\n  gain: {st.stage_gain}\n  norm: {st.normalization_factor:.2e} @ {st.normalization_frequency}Hz\n'
            out = out + expr
    out = out + '\nAll seismometer reponses\n'
    for seism, v in SEISMOMETERS.items():
        if v['rstage']:
            out = out + f"{seism} has ObsPy {v['rstage']}\n"
    if print_:
        print(out)
    with open(OUT + 'expressions_for_table.txt', 'w') as f:
        f.write(out)


def add_seis_response_stages():
    for sm, v in SEISMOMETERS.items():
        gain = v['sensitivity V/m*s(*s)']
        reffreq = v['reffreq (Hz)']
        if v[NRLKEY]:
            resp = NRL.get_sensor_response(v[NRLKEY])
            rstage = resp.response_stages[0]
            if gain:
                rstage.stage_gain = gain
            if reffreq:
                rstage.stage_gain_frequency = reffreq
        elif v['poles']:
            zeros = [0j, 0j]
            if not reffreq:
                reffreq = 1
            gainfreq = normfreq = reffreq
            norm = 1
            stage_number = 1
            rstage = PolesZerosResponseStage(
                stage_number, gain, gainfreq, 'M/S', 'V',
                'LAPLACE (RADIANS/SECOND)',
                normfreq, zeros, v['poles'], norm
                )
        else:
            rstage = None
        v['rstage'] = rstage


def get_digi_response_stages(digitizer, sr):
    v = DIGITIZERS[digitizer]
    gain = v['sensitivity count/V']
    if v[NRLKEY]:
          # digitizer in NRL
          k1_sr = copy(v[NRLKEY])
          k1_sr[-1] = k1_sr[-1].format(sr=sr)
          rstages = NRL.get_datalogger_response(k1_sr).response_stages
          if NRL._nrl_version == 1:
              rstages.pop(0)
          else:
              for stage in rstages:
                  stage.stage_sequence_number += 1
          # digitizer gain different from NRL
          # have to be checked agan for NRLv2
          if gain:
              rstages[1].stage_gain = gain
    elif gain:
        # digitizer not in NRL
        stage_number = 2
        gain_freq = 1
        rstages = [CoefficientsTypeResponseStage(
            stage_number, gain, gain_freq, 'V', 'COUNTS', 'DIGITAL',
            numerator=[1], denominator=[],
            decimation_input_sample_rate=sr,
            decimation_factor=1,
            decimation_offset=0, decimation_delay=0,
            decimation_correction=0
            )]
    else:
        raise ValueError(f'{digitizer}: Neither NRL keys nor gain given')
    return rstages


def add_software_decimation_stages(rstages, sr, sr2, cascade):
    nstage = len(rstages) + 1
    units = rstages[-1].output_units
    reffreq = rstages[-1].stage_gain_frequency
    assert sr % sr2 == 0
    if cascade is None:
        dec_stages = [CoefficientsTypeResponseStage(
            nstage, 1, reffreq, units, units, 'DIGITAL',
            numerator=[1], denominator=[],
            decimation_input_sample_rate=sr,
            decimation_factor=sr // sr2,
            decimation_offset=0, decimation_delay=0,
            decimation_correction=0
            )]
        sr = sr2
    else:
        dec_stages = []
        for i, lp in enumerate(cascade.split('-')):
            filt = DECIMATION_FILTERS[lp]
            coeffs = filt['coefficients']
            if filt['symmetry'] == 'EVEN':
                coeffs = coeffs + coeffs[::-1]
            else:
                # SeisComp Warning: The coefficients for non-symmetric (minimum-phase)
                # FIR filters in the filters.fir file are stored in reverse order.
                coeffs = coeffs[::-1]
                raise NotImplementedError()
            assert len(coeffs) == filt['number']
            assert sr % filt['decimation'] == 0
            st = CoefficientsTypeResponseStage(
                nstage + i, 1, reffreq, units, units, 'DIGITAL',
                name=lp, numerator=coeffs, denominator=[],
                decimation_input_sample_rate=sr,
                decimation_factor=filt['decimation'],
                decimation_offset=0, decimation_delay=0,
                decimation_correction=0
                )
            dec_stages.append(st)
            sr = sr // filt['decimation']
    assert sr == sr2
    rstages.extend(dec_stages)


def csv2xml(only_public=False):
    """Write StationXML, SH sensitivity and statinf"""
    stations = []
    shm_sens = []
    shm_loc = []
    info = []
    warned = None
    # tsn is dictionary {station_code: epoch_list}
    tsn = csv2dictlist(NET_CODE)
    # sort epochs by time, coordinates of latest epoch will be used as station
    # coordinates
    for sta in tsn:
        tsn[sta] = sorted(tsn[sta], key = lambda epoch: epoch['UTC_starttime'])
    for sta in tqdm(tsn):
        sta_startdate = None
        sta_coords = None
        channels = []
        for epoch in tsn[sta]:
            fc = 0
            if only_public and epoch['public'] != 'TRUE':
                continue
            sta_code = epoch['station code']
            loc_cha_code = _setdefault(epoch, 'location and channel (non default)', DEFAULT_LOC_CHA_CODE)
            loc_code, cha_code_template = loc_cha_code.split('.')
            srs = _setdefault(epoch, 'sampling rates', DEFAULT_SRS)
            srs = srs.replace(' ', '').split(',')
            startdate = UTC(epoch['UTC_starttime'])
            enddate = UTC(epoch['UTC_endtime']) if epoch['UTC_endtime'] != '' else None
            _setdefault(epoch, 'seismometer id')
            if sta_startdate is None:
                sta_startdate = startdate
            try:
                lat = float(epoch['latitude'])
                lon = float(epoch['longitude'])
                elev = float(epoch['elevation'])
                depth = float(epoch['depth'])
                digi = DIGITIZERS[epoch['digitizer']]
                seism = SEISMOMETERS[epoch['seismometer type']]
            except (ValueError, KeyError):
                msg = ('{station code} {UTC_starttime} -- Ignore epoch, '
                       'problem with coordinates, seismometer or digitizer')
                print(msg.format(**epoch))
                continue
            if sta_coords is None:
                sta_coords = (lat, lon, elev)
            elif sta_coords != 'invalid' and sta_coords != (lat, lon, elev):
                sta_coords = 'invalid'
                print(f'{sta_code} is moving! Coordinates at station level will reflect latest epoch.')
            data_logger = Equipment(manufacturer=digi[NRLKEY][0] if digi[NRLKEY] else None,
                                    description=epoch['digitizer'],
                                    model=epoch['digitizer'].split('_')[0])
            sensor = Equipment(manufacturer=seism[NRLKEY][0]  if seism[NRLKEY] else None,
                               description=epoch['seismometer type'],
                               model=epoch['seismometer type'],
                               serial_number=epoch['seismometer id'])
            # create response_stage for seismometer
            if seism['rstage'] is None:
                print(f"{sta_code} No response for seismometer {epoch['seismometer type']}")
                continue
            for stream in srs:
                if '->' in stream:
                    if ':' in stream:
                        decimate, cascade = stream.split(':')
                    else:
                        decimate, cascade = stream, None
                    sr, sr2 = map(int, decimate.split('->'))
                else:
                    sr = sr2 = int(stream)
                    decimate = None

                ###  create response object
                digi_stages = get_digi_response_stages(epoch['digitizer'], sr)
                rstages = [seism['rstage']] + digi_stages
                if decimate:
                    add_software_decimation_stages(rstages, sr, sr2, cascade)
                if sr2 >= 2:
                    reffreq = seism['reffreq (Hz)'] or 1.
                else:
                    reffreq = 0.3
                instrument_sensitivity = InstrumentSensitivity(
                    1, 1,
                    input_units = seism['rstage'].input_units,
                    input_units_description = seism['rstage'].input_units_description,
                    output_units = digi_stages[-1].output_units,
                    output_units_description = digi_stages[-1].output_units_description
                    )
                response = Response(response_stages=rstages, instrument_sensitivity=instrument_sensitivity)
                response.recalculate_overall_sensitivity(reffreq)
                ###

                sum_sens = 0
                if fc == 0 and response.instrument_sensitivity.input_units == 'M/S':
                    fc = calc_fc(response)
                if len(cha_code_template) == 3:
                    comps = cha_code_template[-1]
                    warn = f"{sta_code} {epoch['UTC_starttime']} {cha_code_template}: Epoch defines only one component"
                    if warned != warn:
                        print(warn)
                        warned = warn
                    cha_code_template2 = cha_code_template[:-1]
                else:
                    cha_code_template2 = cha_code_template
                    comps = 'ZNE'
                for comp in comps:
                    # cha_code = cha_code_template.replace('?', SR2CODE[sr2]) + comp
                    # seed_id = '.'.join([NET_CODE, sta_code, loc_code, cha_code])
                    # path1 = f'{RESP}{seed_id}_{startdate!s:.10}.resp'
                    # path2 = f'{RESP}{seed_id}.resp'
                    # path = path1 if os.path.exists(path1) else path2
                    # if os.path.exists(path):
                    #     print(f'  Found and use response file at {path}')
                    #     chresponse = read_inventory(path)[0][0][0].response
                    #     chresponse.recalculate_overall_sensitivity(reffreq)
                    # adjust gain in response
                    try:
                        custom_compgain = SEISMOMETERS_CUSTOM_GAIN_COMP[epoch['seismometer id']]
                    except KeyError:
                        chresponse = response
                    else:
                        chresponse = deepcopy(response)
                        chresponse.response_stages[0].stage_gain = custom_compgain['ZNE'.index(comp)]
                        chresponse.recalculate_overall_sensitivity(reffreq)
                    sum_sens += chresponse.instrument_sensitivity.value
                    cha_code = cha_code_template2.replace('?', SR2CODE[sr2]) + comp
                    # create channel
                    kw = dict(code=cha_code, location_code=loc_code,
                              latitude=lat, longitude=lon,
                              elevation=elev-depth, depth=depth,
                              azimuth=COMP2AZIDIP[comp][0],
                              dip=COMP2AZIDIP[comp][1],
                              sample_rate=sr2, response=chresponse,
                              start_date=startdate, end_date=enddate,
                              sensor=sensor, data_logger=data_logger)
                    channels.append(Channel(**kw))
                    fargs = (sta_code.lower(), cha_code.lower()[:2], comp.lower(),
                             _date2sh(startdate), _date2sh(enddate),
                             1e9 / chresponse.instrument_sensitivity.value)
                    if epoch['seismometer type'] != 'CMG-5T':  # ignore accelerometers
                        shm_sens.append(SHM_SENS.format(*fargs))
                # check that sensitivity values in table are correct within 1%
                sens = sum_sens / len(comps)
                sens_table = float(epoch['count/m*s(*s)'])
                error = abs(sens_table - sens) / sens
                if error > 0.01:
                    print('{station code} {UTC_starttime} -- '.format(**epoch) +
                          'error in sensitivity reported in TSN table, loc.cha {}.{}, expected: {:.3e}, got: {:.3e}, difference: {:.2f}%'.format(loc_code, cha_code, sens, sens_table, error * 100))
                    # print(response)
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
                 fc, chresponse.get_paz().normalization_factor,
                 chresponse.get_paz().zeros, chresponse.get_paz().poles)
        info.append(INFO.format(*fargs))
        stations.append(Station(code=sta_code, latitude=lat, longitude=lon,
                      elevation=elev, creation_date=sta_startdate,
                      restricted_status='open',
                      site=Site(name=CONF.get('STA_NAME', '{}').format(name),
                                country='Germany'),
                      channels=channels,
                      start_date=sta_startdate, end_date=enddate))
    net = Network(code=NET_CODE, stations=stations,
                  description=CONF.get('NET_DESC'),
                  start_date=UTC(CONF['NET_START']), restricted_status='open',
                  identifiers=CONF.get('IDENTIFIERS'))
    inv = Inventory(networks=[net], source=CONF.get('SOURCE'))
    fname = NET_CODE + '_private' * (not only_public)
    inv.write(OUT + fname + '.xml', 'STATIONXML', validate=True)
    inv.write(OUT + fname + '.txt', 'STATIONTXT', validate=True)
    with open(OUT + 'SH_' + fname + '_statinf.dat', 'w') as f:
        f.write('\n'.join(shm_loc))
    with open(OUT + 'SH_' + fname + '_sensitivities.txt', 'w') as f:
        f.write('\n'.join(shm_sens))
    print()
    print('PUBLIC: ', only_public)
    print(INFO_INFO)
    print('\n'.join(info))
    print()


parser = argparse.ArgumentParser(description='Create StationXML from csv files')
parser.add_argument('-c', '--conf', default='conf.json', help='Name of config file (default: conf.json)')
parser.add_argument('--pdb', action='store_true', help='Start the debugger upon exception')
parser.add_argument('-o', '--only', choices=['public', 'private', 'info'], help='Only do some stuff')
cliargs = parser.parse_args()
if cliargs.pdb:
    import pdb
    import sys
    import traceback
    def info(type, value, tb):
        traceback.print_exception(type, value, tb)
        print()
        pdb.pm()
    sys.excepthook = info
with open(cliargs.conf) as f:
    CONF = json.load(f, cls=ConfigJSONDecoder)
OUT = CONF.get('OUT', '')
NET_CODE = CONF['NET_CODE']
NRL = NRLClient(CONF['NRLPATH']) if CONF.get('NRLPATH') is not None else NRLClient()
NRLKEY = CONF.get('NRLKEY', 'NRLv1')

DEFAULT_LOC_CHA_CODE = '.?H'
DEFAULT_SRS = '100'
SR2CODE = {1000: 'G', 400: 'D', 200: 'H', 100: 'H', 20: 'B', 1: 'L'}
COMP2AZIDIP = {'Z': (0, -90), 'N': (0, 0), 'E': (90, 0)}

SHM_LOC = '{:5}  lat:{:+.6f}  lon:{:+.6f}  elevation:{:.1f}  name:{}'
SHM_SENS = '{}-{}-{} {} {} {:.5f}'
INFO_INFO = 'code  name                      lat     lon     elev   gain      SH    PAZ'
INFO = '{:5} {:25} {:.3f}  {:.3f}  {:.1f}  {:.2e}  {:.2f}  fc:{:.3f}Hz norm:{:.2e} zeros:{} poles:{}'

(
    SEISMOMETERS,
    SEISMOMETERS_CUSTOM_GAIN_COMP,
    DIGITIZERS,
    DECIMATION_FILTERS
) = load_tables()
add_seis_response_stages()

if not cliargs.only or cliargs.only == 'info':
    write_info(print_=(cliargs.only == 'info'))
if not cliargs.only or cliargs.only == 'public':
    csv2xml(only_public=True)
if not cliargs.only or cliargs.only == 'private':
    csv2xml(only_public=False)

# Warning:
# UserWarning: More than one PolesZerosResponseStage encountered. Returning first one found.
# is raised, because Centaur datalogger has an additional PAZ defined.
# Can be ignored.
