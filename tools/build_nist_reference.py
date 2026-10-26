"""Build an authoritative vacuum line reference from the NIST ASD (Ritz energies: λ_vac = 1e8/(Ek−Ei) Å)."""
import csv
import io
import sys
import urllib.parse
import urllib.request

OUT = sys.argv[1]
# species and vacuum-ish wavelength windows (Å) relevant to UCD RVs, optical to K band
QUERIES = [
    ('H I', 6500, 6600), ('Li I', 6700, 6720), ('Na I', 5880, 5900), ('Na I', 8170, 8210), ('Na I', 11370, 11420),
    ('Na I', 22000, 22120), ('K I', 7650, 7710), ('K I', 11680, 11790), ('K I', 12420, 12540), ('K I', 15150, 15180),
    ('Rb I', 7790, 7960), ('Cs I', 8510, 8960), ('Fe I', 11600, 12000), ('Al I', 13100, 13170),
    ('Ca I', 19440, 19870), ('Mg I', 15020, 15050), ('Ti I', 9600, 9800),
]
BASE = 'https://physics.nist.gov/cgi-bin/ASD/lines1.pl'


def query(spec, lo, hi):
    # NIST windows are in air above 2000 Å; widen by 4 Å so vacuum targets are covered
    params = dict(spectra=spec, output_type=0, low_w=lo - 4, upp_w=hi + 4, unit=0, submit='Retrieve Data', de=0,
                  format=3, line_out=0, remove_js='on', en_unit=0, output=0, bibrefs=1, page_size=15, show_obs_wl=1,
                  show_calc_wl=1, unc_out=1, order_out=0, show_av=2, tsb_value=0, A_out=0, intens_out='on',
                  allowed_out=1, forbid_out=1, conf_out='on', term_out='on', enrg_out='on', J_out='on')
    url = BASE + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': 'curl/8.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        text = r.read().decode()
    rows = list(csv.DictReader(io.StringIO(text), delimiter='\t'))
    out = []
    for row in rows:
        clean = {k: (v or '').strip('"') for k, v in row.items() if k}
        try:
            ei, ek = float(clean['Ei(cm-1)'].strip('[]()')), float(clean['Ek(cm-1)'].strip('[]()'))
        except (KeyError, ValueError):
            continue
        if ek <= ei:
            continue
        wvac = 1e8 / (ek - ei)
        if not (lo - 6 <= wvac <= hi + 6):
            continue
        out.append(dict(species=spec, wave_vac=round(wvac, 4), wave_air=clean.get('ritz_wl_air(A)', ''),
                        acc=clean.get('Acc', ''), aki=clean.get('Aki(s^-1)', ''), intens=clean.get('intens', ''),
                        lower=f"{clean.get('conf_i', '')} {clean.get('term_i', '')} J={clean.get('J_i', '')}",
                        upper=f"{clean.get('conf_k', '')} {clean.get('term_k', '')} J={clean.get('J_k', '')}",
                        ei=ei, ek=ek, ref=clean.get('line_ref', '') or clean.get('tp_ref', '')))
    return out


allrows = []
for spec, lo, hi in QUERIES:
    rows = query(spec, lo, hi)
    print(f'{spec:5s} {lo}-{hi}: {len(rows)} lines', file=sys.stderr)
    allrows += rows
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(allrows[0]))
    w.writeheader()
    w.writerows(allrows)
print(f'wrote {len(allrows)} lines to {OUT}', file=sys.stderr)
