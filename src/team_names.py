"""
Club-name resolution between data sources.
==========================================
OddsAPI (used at PREDICT time) and football-data.co.uk (the HISTORY we train on) spell the
same club very differently:

    OddsAPI                     football-data.co.uk
    1. FC Kaiserslautern        Kaiserslautern      ordinal + generic prefix
    VfL Wolfsburg               Wolfsburg           generic prefix
    SV Darmstadt 98             Darmstadt           generic prefix + year suffix
    Arminia Bielefeld           Bielefeld           extra descriptor token
    Lincoln City                Lincoln             extra descriptor token
    Cadiz CF / Almeria          Cadiz / Almeria     accents + generic suffix
    Karlsruher SC               Karlsruhe           inflected stem
    Sporting Gijon              Sp Gijon            abbreviated token
    Queens Park Rangers         QPR                 initialism (alias table)

`build_features` (training) reads football-data names; `build_upcoming_features` (predict)
receives OddsAPI names. When they fail to reconcile, the fixture gets NO rolling-form
features and every one of them is median-imputed — the model is effectively blind on that
match. Measured 2026-08-15: 46% of standard fixtures, 27 clubs, on the core features
(scored/conceded/over25 rate/attack & defense strength), not the 0%-importance extras.

SAFETY: the old fallback was `history_name.lower().startswith(oddsapi_name.split()[0])`,
which would happily map "Real Valladolid CF" onto any club beginning with "Real". Here,
resolution is always scoped to ONE league and an ambiguous match — more than one candidate
at the same confidence level — is REJECTED rather than guessed. Precision over recall: a
missing club costs one fixture's features, a wrong club poisons them with another team's form.
"""
from __future__ import annotations

import re
import unicodedata

# Club-form words that carry no identity on their own.
_GENERIC = {
    "fc", "cf", "sc", "afc", "cd", "sd", "ad", "ac", "ss", "as", "us", "sv", "vfl", "vfb",
    "fsv", "tsg", "spvgg", "msv", "bsc", "if", "ifk", "fk", "sk", "bk", "ca", "cp", "club",
    "calcio", "sporting",  # "Sporting Gijon" -> Gijon; the alias table covers Sporting CP
    # Nordic / Belgian / German club-form abbreviations (Fotbollforening, Idrottsklubb,
    # Koninklijke Voetbalclub, Turn- und Sportverein ...), added 2026-11-02 with the stem tiers'
    # removal: "Kalmar FF" == "Kalmar", "KV Mechelen" == "Mechelen", "TSV Hartberg" == "Hartberg"
    "ff", "ik", "aif", "kv", "kvc", "tsv", "ssv", "rc", "bc",
    # the club-form affixes data_loader._norm_club stripped, so identity_key() (which replaces it in
    # data_loader._canonicalise_clubs) keeps its merges: "UD Almeria" == "Almeria",
    # "SSC Napoli" == "Napoli", "NK Osijek" == "Osijek" (2026-11-02)
    "ud", "nk", "vv", "ssc", "usc", "cfc",
    "real",  # dropped only as a LAST resort, see _tokens(keep_real)
}
# Descriptors that distinguish English clubs from each other and must NEVER be stripped
# globally ("Manchester City" vs "Manchester United"). Subset matching handles them instead.
_KEEP = {"city", "united", "utd", "town", "rovers", "rvs", "county", "athletic", "wanderers"}
# One word written two ways (football-data contracts these). Spelling the word out is text
# normalisation, not an identity guess: "Dundee Utd" == "Dundee United", never "Dundee".
_ABBREV = {"utd": "united", "rvs": "rovers"}

_ALIASES = {
    # initialisms and heavy abbreviations no token rule can derive
    "queens park rangers": "qpr",
    # football-data contracts by dropping vowels, which no stem rule can derive
    "bristol rovers": "bristol rvs",
    "blackburn rovers": "blackburn",
    "tranmere rovers": "tranmere",
    "doncaster rovers": "doncaster",
    "forest green rovers": "forest green",
    "peterborough united": "peterboro",
    "sporting gijon": "sp gijon",
    "sheffield wednesday": "sheffield weds",
    "west bromwich albion": "west brom",
    "wolverhampton wanderers": "wolves",
    "brighton and hove albion": "brighton",
    "milton keynes dons": "mk dons",
    "deportivo la coruna": "la coruna",
    "racing santander": "santander",
    "real sporting de gijon": "sp gijon",
    # German endonym/exonym pairs no token or stem rule can bridge. Added 2026-08-17 when
    # prop-odds matching showed Bayern and Koln failing against OddsAPI's English spellings.
    "bayern munich": "bayern munchen",
    "cologne": "koln",
    # OddsAPI spellings that resolved to NOTHING, so the fixture got no form data at all
    # (invariant 8). Owner-approved 2026-11-01 ("we must fix those bugs"). MLS: one history spelling
    # each, so this only adds data. China: the history holds each club under TWO names with
    # overlapping dates (a duplicate identity, the Standard bug the NF repair still has to fix);
    # until then the CURRENT name is used.
    "d c united": "dc united",
    "la galaxy": "los angeles galaxy",
    "shandong luneng taishan fc": "shandong taishan",
    "henan fc": "henan songshan longmen",
    # Data Truth audit 2026-11-01 (finding 6): each club has ONE spelling in its league's history
    # (checked on fd_history + af_history since 2024), so these only add data - the clubs had no
    # form at all and every fixture of theirs was AVOID as blind. Clubs whose history holds TWO
    # spellings (Belgrano, Kyoto, Urawa, UNAM, Celta B) are NOT aliased here: that needs the
    # duplicate identity unified first (finding 37), or an alias would drop half their history.
    "saint etienne": "st etienne",
    "austria wien": "austria vienna",
    "atletico mineiro": "atletico mg",
    "argentinos juniors": "argentinos jrs",
    # Data Truth repair 2026-11-01 (data profile: unresolved_club_name). Same rule as above: ONE
    # spelling of the club in its league's history bank (fd_history + af_history), and the same
    # club proven at fixture level - on every ledger/quote fixture of the predict spelling, the
    # bank row for that league, date (+-1 day) and the resolved opponent carries exactly this
    # history spelling in the club's slot (counts in brackets).
    "estudiantes": "estudiantes l p",           # Argentina; Estudiantes de Rio Cuarto is another club (27/27)
    "gimnasia la plata": "gimnasia l p",        # Argentina; Gimnasia Mendoza is another club (26/29)
    "cercle brugge ksv": "cercle brugge",       # Belgium; tokens also hit Club Brugge (2/2)
    "oh leuven": "oud heverlee leuven",         # Belgium (4/4)
    "beijing fc": "beijing guoan",              # China; Beijing Renhe last played 2019 (24/24)
    "boston united": "boston utd",              # National League (6/6)
    "inverness ct": "inverness c",              # Scottish Championship (6/6)
    "raith rovers": "raith rvs",                # Scottish Championship (4/4)
    "airdrie united": "airdrie utd",            # Scottish League One (4/4)
    "queen of the south": "queen of sth",       # Scottish League One (4/4)
    "sporting cp": "sp lisbon",                 # Portugal (4/4)
    "vitoria sc": "guimaraes",                  # Portugal; Vitoria Setubal is "Setubal" (4/4)
    "basaksehir": "buyuksehyr",                 # Turkey (6/6)
    # Moved here from src/api_football_ou._POST_NORM_ALIASES (2026-11-02), so every join uses
    # the one alias store. API-Football / football-data spellings of the same club.
    "vasco da gama": "vasco",
    "rb bragantino": "bragantino",
    "red bull bragantino": "bragantino",
    "atlanta united": "atlanta utd",
    "internazionale": "inter",
    "sporting": "sporting cp",
    # 2026-11-02: the stem and token-subset tiers were removed (they paired different clubs). Every
    # pair those tiers used to bridge on the repo's own data was re-checked AT FIXTURE LEVEL: on
    # each predict/quote/props fixture of the predict spelling (predictions, ledgers, odds
    # captures, player_prop_odds_history), the history row of that league, date (+-1 day) and the
    # resolved opponent carries this spelling in the club's slot (fixture counts in brackets;
    # pools: fd_history + af_history, and player_history for API-Football spellings). A tuple
    # lists every spelling the history holds for the one club; a pool holding two of them is a
    # duplicate identity and resolve() refuses it instead of picking half the history.
    # The old tiers were WRONG for 'Independiente Rivadavia' (-> 'Independiente'), 'Shenzhen
    # Peng City FC' (-> 'Shenzhen') and 'Dundee Utd' (-> 'Dundee'); the entries below are the
    # clubs the fixtures show.
    # Argentina
    "aldosivi mar del plata": "aldosivi",                       # (25)
    "atl tucuman": "atletico tucuman",                          # (8)
    "atletico huracan": "huracan",                              # (25)
    "belgrano de cordoba": ("belgrano", "belgrano cordoba"),    # (11) two history spellings
    "ca tigre ba": "tigre",                                     # (27)
    "central cordoba": "central cordoba de santiago",           # (22)
    "dep riestra": "deportivo riestra",                         # (16)
    # no predict fixture in the history yet; the pair is in the audited identity map
    # (config/standard_ou25_identity_aliases.json, Argentina)
    "estudiantes rio cuarto": "estudiantes de rio cuarto",
    "independiente rivadavia": ("ind rivadavia", "independ rivadavia"),   # (7 + 15)
    "instituto de cordoba": ("instituto", "instituto cordoba"),  # (10) two history spellings
    "sarmiento de junin": "sarmiento junin",                    # (28)
    "talleres": "talleres cordoba",                             # (24)
    "union de santa fe": "union santa fe",                      # (8)
    "velez sarsfield ba": "velez sarsfield",                    # (22)
    # Austria (two spellings each in the history)
    "lask": "lask linz",                                        # (37)
    "rapid wien": ("rapid vienna", "sk rapid"),                 # (3 + 6; SK Rapid -> Rapid Vienna 28)
    "rb salzburg": ("red bull salzburg", "salzburg"),           # (3 + 9; Salzburg -> RB Salzburg 30)
    "rheindorf altach": ("altach", "scr altach"),               # (5 + 2; Altach -> SCR Altach 28)
    "wsg tirol": ("tirol", "wsg wattens"),                      # (7 + 2)
    # Belgium
    "lommel united": "lommel sk",                               # (4)
    "standard liege": "standard",                               # (4)
    "zulte waregem": "waregem",                                 # (5)
    "sv zulte waregem": "waregem",                              # (1)
    "union st gilloise": "st gilloise",                         # (4)
    # Brazil
    "bragantino sp": ("bragantino", "rb bragantino"),           # (7 + 8; Bragantino -> RB 24)
    "botafogo rj": "botafogo",                                  # (26)
    "flamengo rj": "flamengo",                                  # (21)
    "fortaleza": "fortaleza ec",                                # (25)
    # Bundesliga 2
    "bielefeld": "arminia bielefeld",                           # (32)
    "braunschweig": "eintracht braunschweig",                   # (65)
    "dresden": "dynamo dresden",                                # (32)
    "hamburg": "hamburger sv",                                  # (33)
    "karlsruhe": "karlsruher sc",                               # (65)
    # England
    "birmingham city": "birmingham",                            # Championship (24)
    "bolton wanderers": "bolton",                               # Championship / League One (26)
    "cardiff city": "cardiff",                                  # (24)
    "charlton athletic": "charlton",                            # (24)
    "derby county": "derby",                                    # (24)
    "hull": "hull city",                                        # (88)
    "lincoln city": "lincoln",                                  # (24)
    "norwich city": "norwich",                                  # (24)
    "oxford": "oxford united",                                  # (88)
    "preston north end": "preston",                             # (24)
    "stoke": "stoke city",                                      # (88)
    "swansea city": "swansea",                                  # (24)
    "west ham united": "west ham",                              # (24)
    "bradford city": "bradford",                                # League One (24)
    "burton": "burton albion",                                  # (92)
    "cambridge": "cambridge united",                            # League One / Two (92)
    "exeter": "exeter city",                                    # (92)
    "huddersfield town": "huddersfield",                        # (24)
    "leicester city": "leicester",                              # (24)
    "mansfield": "mansfield town",                              # (91)
    "peterboro": "peterborough",                                # (92)
    "plymouth argyle": "plymouth",                              # (26)
    "rotherham united": "rotherham",                            # (28)
    "stockport": "stockport county",                            # (92)
    "wigan athletic": "wigan",                                  # (26)
    "wycombe wanderers": "wycombe",                             # (26)
    "accrington stanley": ("accrington", "accrington st"),      # League Two (9 + 4; Accrington -> ST 92)
    "cheltenham town": "cheltenham",                            # (25)
    "colchester united": "colchester",                          # (23)
    "crewe alexandra": "crewe",                                 # (26)
    "grimsby town": "grimsby",                                  # (25)
    "harrogate town": "harrogate",                              # (9 + 92)
    "northampton town": "northampton",                          # (22)
    "oldham athletic": "oldham",                                # (23)
    "salford": "salford city",                                  # (92)
    "shrewsbury town": "shrewsbury",                            # (26)
    "swindon": "swindon town",                                  # (91)
    "york city": "york",                                        # (25)
    "aldershot town": "aldershot",                              # National League (9)
    "fc halifax town": "halifax",                               # (9)
    "kidderminster harriers": "kidderminster",                  # (9)
    "solihull moors": "solihull",                               # (9)
    "sutton utd": "sutton",                                     # (9)
    "yeovil town": "yeovil",                                    # (9)
    "coventry city": "coventry",                                # Premier League (1)
    "ipswich town": "ipswich",                                  # (1)
    "leeds united": "leeds",                                    # (1)
    "tottenham hotspur": "tottenham",                           # (1)
    # FPL short names (output/fpl_bootstrap.json) vs API-Football, matched by FPL team id on
    # output/fpl_fixtures.json kickoff dates against player_history Premier League rows
    "man city": "manchester city",                              # (8)
    "man utd": "manchester united",                             # (7)
    "nott m forest": "nottingham forest",                       # (8)
    "spurs": "tottenham",                                       # (7)
    # China
    "chongqing tonglianglong": "chongqing tongliang long",      # (1)
    "chongqing tonglianglong fc": "chongqing tongliang long",   # (6)
    "shenzhen peng city fc": ("shenzhen xinpengcheng", "sichuan jiuniu"),  # (7 + 5)
    "zhejiang": ("hangzhou greentown", "zhejiang professional"),           # (5 + 8)
    # Denmark / Netherlands / Finland / Norway / Sweden
    "agf aarhus": "aarhus",                                     # (22)
    "ob odense bk": "odense",                                   # (22)
    "ado den haag": "den haag",                                 # (6)
    "fortuna sittard": "for sittard",                           # (6)
    "nec nijmegen": "nijmegen",                                 # (7)
    "pec zwolle": "zwolle",                                     # (5)
    "hjk": "hjk helsinki",                                      # (18)
    "ilves tampere": "ilves",                                   # (15)
    "kups kuopio": "kups",                                      # (25)
    "sjk seinajoki": "sjk",                                     # (22)
    "tps turku": ("tps", "turku ps"),                           # (4 + 9)
    "vps vaasa": "vps",                                         # (27)
    "hamkam": "ham kam",                                        # (32)
    "kfum": "kfum oslo",                                        # (14)
    "sarpsborg fk": ("sarpsborg 08", "sarpsborg 08 ff"),        # (3 + 6)
    "aik": "aik stockholm",                                     # (30)
    "djurgarden": "djurgardens if",                             # (18)
    "halmstads bk": "halmstad",                                 # (22)
    "hammarby if": ("hammarby", "hammarby ff"),                 # (7 + 8)
    "oster": "osters if",                                       # (18)
    # Ireland
    "drogheda": "drogheda united",                              # (30)
    "galway": "galway united",                                  # (31)
    "shelbourne dublin": "shelbourne",                          # (27)
    "st patricks athletic": ("st patrick s athl", "st patricks"),  # (14; St. Patricks -> Athl. 31)
    # Japan
    "kashima antlers": "kashima",                               # (27)
    "kyoto purple sanga": ("kyoto", "kyoto sanga"),             # (13) two history spellings
    "machida": "machida zelvia",                                # (12)
    "okayama": "fagiano okayama",                               # (12)
    "urawa red diamonds": ("urawa", "urawa reds"),              # (12) two history spellings
    "verdy": "tokyo verdy",                                     # (13)
    # Spain / France / Italy / Portugal / Scotland / Turkey / Mexico
    "ferrol": "racing ferrol",                                  # La Liga 2 (42)
    "clermont": "clermont foot",                                # Ligue 2 (77 + 3)
    "rodez af": "rodez",                                        # (17)
    "stade de reims": "reims",                                  # (16)
    "stade lavallois": "laval",                                 # (17)
    "troyes": "estac troyes",                                   # (68)
    "usl dunkerque": "dunkerque",                               # (17)
    "vicenza": "vicenza virtus",                                # Serie B (10)
    "sc braga": "sp braga",                                     # Portugal (6)
    "ayr utd": "ayr",                                           # Scottish Championship (6)
    "queen s park": "queens park",                              # (7)
    "alloa athletic": "alloa",                                  # Scottish League One (2)
    "hamilton academical": "hamilton",                          # (3)
    "elgin city": "elgin",                                      # Scottish League Two (1)
    "forfar athletic": "forfar",                                # (1)
    "stirling albion": "stirling",                              # (1)
    "heart of midlothian": "hearts",                            # Scottish Premiership (7)
    "amed": "amedspor",                                         # Turkey (6)
    "caykur rizespor": "rizespor",                              # (1)
    "genclerbirligi s k": "genclerbirligi",                     # (6)
    "goztepe": "goztep",                                        # (6)
    "torku konyaspor": "konyaspor",                             # (1)
    "atl san luis": "atletico san luis",                        # Liga MX (16)
    "guadalajara": "guadalajara chivas",                        # (22)
    "pumas": ("u n a m pumas", "unam pumas"),                   # (13) two history spellings
    "tigres": "tigres uanl",                                    # (25)
    # 2026-11-02 (data5): API-Football gives one club two spellings - the /fixtures side name and
    # the /fixtures/players team-block name. FIXTURE-LEVEL proof in player_history.parquet: on every
    # fixture_id carrying the block name, the fixture's side is the alias spelling, the other
    # block names the other side, and the two spellings never meet (fixtures/rows in brackets).
    # Without them 1,122 player_history rows could not be put on a side (own club as opponent).
    "hellas verona": "verona",                  # Serie A (114 fixtures / 1,687 rows); fd_history
                                                # Serie B 2026-08-23 / -29: same date, score, opponent
    "cf montreal": "montreal impact",           # USA MLS (36 / 529)
    "czechia": "czech republic",                # WC Qualification Europe (3 / 34)
    # Romanian Superliga: 'CS Universitatea Craiova' (API-Football source, to 2022-05) and
    # 'Univ. Craiova' (football-data, from 2022-01) hold the SAME 20 matches of 2022-01-22..05-27
    # in fd_history: same date, same score, the opponent the same club (10 identical strings, 10
    # spelled differently: 'Rapid' / 'FC Rapid Bucuresti'), and the two names never meet.
    # 'U Craiova 1948' (FC U Craiova 1948) is a DIFFERENT club: on 2022-02-14 both copies read
    # Craiova 2-0 U Craiova 1948. 'Universitatea Craiova' is the af/API spelling from 2022-07.
    "cs universitatea craiova": ("univ craiova", "universitatea craiova"),
    # Fix round 9 (owner 2026-11-04, KEY=bank): identity gaps that dropped af_history / sidecar rows
    # and left board clubs blind. FIXTURE-LEVEL proof against the PROVIDER fixture lists
    # (output/strength/competition_fixtures.parquet: fixture id, team id, FT): every row of each
    # spelling in fd_history / af_history (provider rows) / the committed ledgers was placed on the
    # provider fixture of the same league, date +-1 and strictly resolved opponent, and the club's
    # slot carries ONE provider team id for every spelling; FT agrees on every scored row unless
    # noted. Counts: rows placed on that team id per spelling (history rows; ledger fixtures).
    # The two spellings never meet in one row. Same pairs the loader's same-day identity step
    # derived (IDENTITY_ALIASES), now provable by the resolver too.
    "austria lustenau": "a lustenau",           # Austrian Bundesliga, team 1399: 'Austria Lustenau' 77
                                                # (+10 ledger), 'A. Lustenau' 73 (72 FT agree, 1 differs)
    "orgryte is": "orgryte",                    # Sweden Allsvenskan, team 2166: 29 / 26
    "uc dublin": "ucd",                         # Ireland Premier Division, team 3844: 73 / 112
    "atletico paranaense": "athletico pr",      # Brazil Serie A, team 134: 257 (+13 ledger) / 129
    "atletico go": "atletico goianiense",       # Brazil Serie A, team 144: 66 / 150
    "america mg": "america mineiro",            # Brazil Serie A, team 125: 66 / 112
    # board spellings (OddsAPI) with no history spelling of their own; the club's history spellings
    # are already one group above ('machida' / 'machida zelvia')
    "fc machida zelvia": "machida zelvia",      # Japan J-League, team 303: 13 ledger fixtures;
                                                # 'Machida Zelvia' 100, 'Machida' 82
    "shanghai sipg fc": ("shanghai port", "shanghai sipg"),  # China Super League, team 836: 6 ledger
                                                # fixtures; 'SHANGHAI SIPG' 193, 'Shanghai SIPG' 31,
                                                # 'Shanghai Port' 64 (91 with an opponent the lists
                                                # spell differently: not placed, not contradicting)
}


# RESULT-SETTLEMENT ALIASES ONLY (owner-approved 2026-10-22) - used by resolve_result(), NEVER by resolve().
# OddsAPI / predict names -> API-Football RESULT spellings, found by resolving every club in 689 committed
# predictions snapshots against API-Football's own names; without them those clubs' results could not be
# settled (v91 shadow). Kept OUT of _ALIASES on purpose: the training history holds both spellings of some
# clubs (e.g. "SK Rapid" and "Rapid Vienna"), and an alias there would switch which history rows feed the
# model's form features at predict time - a model-input change nobody approved.
_RESULT_ALIASES = {
    # England / Spain
    "sheffield united": "sheffield utd",
    "accrington stanley": "accrington st",
    "celta fortuna": "celta de vigo ii",
    "real sociedad b": "real sociedad ii",
    # Brazil / USA / Finland
    "bragantino sp": "rb bragantino",
    "atletico mineiro": "atletico mg",
    "d c united": "dc united",
    "la galaxy": "los angeles galaxy",
    "tps turku": "turku ps",
    # Argentina - "Estudiantes" is explicit because Estudiantes de Rio Cuarto is a different club
    "argentinos juniors": "argentinos jrs",
    "estudiantes": "estudiantes l p",
    "gimnasia la plata": "gimnasia l p",
    "gimnasia mendoza": "gimnasia m",
    # Austria - German endonyms vs API-Football's English/older forms
    "austria wien": "austria vienna",
    "rapid wien": "rapid vienna",
    "rb salzburg": "red bull salzburg",
    "rheindorf altach": "scr altach",
    "wsg tirol": "wsg wattens",
    # China - API-Football keeps the clubs' FORMER names (each pairing is one-to-one across the
    # 16-club league: renamed club -> its previous name)
    "zhejiang": "hangzhou greentown",
    "chengdu rongcheng fc": "chengdu better city",
    "dalian yingbo": "dalian zhixing",
    "liaoning tieren fc": "shenyang urban",
    "qingdao hainiu fc": "qingdao jonoon",
    "qingdao west coast fc": "qingdao youth island",
    "shenzhen peng city fc": "sichuan jiuniu",
    "tianjin jinmen tiger fc": "tianjin teda",
    # France / Japan (Data Truth audit 2026-11-01, finding 20: never resolved, never graded).
    # Urawa is spelled both "Urawa" and "Urawa Reds" in the result sources, so the token tiers
    # refuse it as ambiguous; the alias names the one spelling to use.
    "saint etienne": "st etienne",
    "urawa red diamonds": "urawa reds",
    # Data Truth repair 2026-11-01: clubs whose history bank holds TWO spellings of the one club
    # (a duplicate identity, finding 37), so the token tiers refuse them as ambiguous. Proven the
    # same club at fixture level: both spellings sit in the club's slot on the same league + date
    # + opponent as the predict spelling. Settlement only - in _ALIASES they would pick one half
    # of the club's form history, a model-input change that waits for the identity merge.
    "belgrano de cordoba": "belgrano",          # Argentina: 'Belgrano' 18 / 'Belgrano Cordoba' 20
    "instituto de cordoba": "instituto",        # Argentina: 'Instituto' 19 / 'Instituto Cordoba' 18
    "chapecoense": "chapecoense sc",            # Brazil: 'Chapecoense-SC' 30 / 'Chapecoense-sc' 24
    "sc paderborn": "paderborn",                # Bundesliga 2: 'Paderborn' 2 / 'SC Paderborn 07' 3
    "shanghai sipg fc": "shanghai port",        # China (renamed 2021): 'Shanghai Port' 24 / 'SHANGHAI SIPG' 7
    "st patricks athletic": "st patricks",      # Ireland: 'St. Patricks' 24 / "St Patrick's Athl." 25
    "kyoto purple sanga": "kyoto sanga",        # Japan: 'Kyoto Sanga' 13 / 'Kyoto' 20
    "andorra cf": "andorra",                    # La Liga 2: 'Andorra' 19 / 'FC Andorra' 10
    "cd castellon": "castellon",                # La Liga 2: 'Castellon' 17 / 'Castellón' 8
    "pumas": "unam pumas",                      # Liga MX: 'UNAM Pumas' 21 / 'U.N.A.M. - Pumas' 18
}


_RESERVE_SUFFIX = {"ii", "b", "reserves"}
_RESERVE_ANY = {"u19", "u21", "u23"}


def _is_reserve(s: str) -> bool:
    """A reserve side: "... II" / "... B" / "... Reserves" as the LAST word, or an age group anywhere.
    Last-word only, so first teams such as "B. Monchengladbach" or "Fortuna Dusseldorf" are never
    mistaken for reserves ("Celta Fortuna" is covered by its explicit alias instead)."""
    w = norm(s).split()
    return bool(w) and (w[-1] in _RESERVE_SUFFIX or bool(set(w) & _RESERVE_ANY))


# Letters NFKD does not decompose to ASCII (they would otherwise be DELETED: 'Bodø' -> 'bod',
# 'Kasımpaşa' -> 'kasmpasa'). Transliteration only - the same letter in ASCII.
_TRANSLIT = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss", "ı": "i",
                           "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d", "þ": "th", "œ": "oe"})


def norm(s: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    s = str(s)
    hit = _NORM_CACHE.get(s)
    if hit is None:
        hit = _NORM_CACHE[s] = _norm(s)
    return hit


# Pure functions of one string, called ~1M times per identity pass of the merged history (the
# data-truth rule 5 stoplight spent ~20 s here): memoised, bounded by the club vocabulary.
_NORM_CACHE: dict = {}
_IKEY_CACHE: dict = {}


def _norm(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(s).translate(_TRANSLIT))
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_str = ascii_str.lower().replace("&", " and ")
    ascii_str = re.sub(r"[^a-z0-9 ]", " ", ascii_str)
    return re.sub(r"\s+", " ", ascii_str).strip()


def _tokens(s: str, keep_real: bool = True) -> set[str]:
    """Identity tokens: generic club words and bare numbers (98, 04, 1.) removed, word
    abbreviations spelled out (utd -> united, rvs -> rovers)."""
    generic = _GENERIC - ({"real"} if keep_real else set())
    out = set()
    for t in norm(s).split():
        if t in generic or t.isdigit():
            continue
        out.add(_ABBREV.get(t, t))
    return out


def identity_key(name) -> str | None:
    """The club-identity key of one spelling WITHIN A LEAGUE: two spellings with the same key pass
    resolve()'s token-set tier for each other ("1. FC Koln" / "FC Koln" / "Koln"; "Cadiz CF" /
    "Cadiz"). Reserve sides never share a key with a first team. None when the name carries no
    identity token at all ("Sporting CP" is only club-form words): such a name proves nothing and
    must not be merged with anything by this key."""
    if name is None or (isinstance(name, float) and name != name):
        return None
    s = str(name)
    if s in _IKEY_CACHE:
        return _IKEY_CACHE[s]
    t = _tokens(s)
    k = (("reserve|" if _is_reserve(s) else "") + " ".join(sorted(t))) if t else None
    _IKEY_CACHE[s] = k
    return k


def alias_groups_in(pool) -> list[set]:
    """Groups of >= 2 spellings in `pool` (one league's club names) that the ALIAS STORE names as
    ONE club. Each alias entry carries fixture-level evidence, so a league history holding two of
    its spellings holds one club under two names - a duplicate identity for the loader to unify
    (resolve() refuses such a pool as ambiguous instead of picking half the history)."""
    by_norm: dict = {}
    for p in pool:
        if p is None or (isinstance(p, float) and p != p):
            continue
        by_norm.setdefault(norm(p), set()).add(str(p))
    groups = _alias_groups(id(_ALIASES), _ALIASES)
    seen, out = set(), []
    for n in by_norm:
        g = groups.get(n)
        if not g or n in seen:
            continue
        present = [m for m in g if m in by_norm]
        seen.update(present)
        if len(present) >= 2:
            out.append(set().union(*(by_norm[m] for m in present)))
    return out


def resolve(name: str, candidates: list[str]) -> str | None:
    """Map an OddsAPI club name onto one of `candidates` (history names for the SAME league).

    Returns the matching candidate ONLY when it is provably the same club; None when there is no
    such candidate or more than one. Tiers are tried in order; a token tier needs EXACTLY ONE hit.

    PROVABLY THE SAME CLUB (2026-11-02, owner rule): normalised equality, the alias store, or the
    SAME identity-token set once club-form words ("FC", "SV", "Real" as a last resort) and bare
    numbers are removed and word abbreviations are spelled out. The stem tier and the
    token-subset tier are GONE: they paired different clubs whenever the right one was absent
    from the pool - 'Platense' -> 'River Plate' ('plate' is a stem of 'platense'),
    'Estudiantes' -> 'Estudiantes de Rio Cuarto', 'Beijing FC' -> 'Beijing Renhe',
    'Dundee United' -> 'Dundee'. A spelling no rule can prove goes in the alias store, with
    fixture-level evidence, or stays unresolved.
    """
    if not name or not candidates:
        return None
    n = norm(name)

    by_norm = {}
    for c in candidates:
        by_norm.setdefault(norm(c), c)

    # 1. identical once normalised
    if n in by_norm:
        return by_norm[n]

    # 2. explicit alias, in either direction
    hit = _alias_hit(n, _ALIASES, by_norm)
    if hit:
        return hit

    nt = _tokens(name)
    if not nt:
        return None
    # RESERVE SIDES ONLY MATCH RESERVE SIDES in the token tiers (2026-10-22): "Celta Vigo" (first
    # team) is a token subset of "Celta de Vigo II" and resolved to it, so a first-team fixture could
    # be settled with the reserve team's result. Explicit aliases above are unaffected.
    res = _is_reserve(name)
    cand_tokens = [(c, _tokens(c)) for c in candidates if _is_reserve(c) == res]

    def _only(hits):
        return hits[0] if len(hits) == 1 else None

    # 3. identical identity-token sets  ("Cadiz CF" == "Cadiz", "Wimbledon" == "AFC Wimbledon")
    hit = _only([c for c, t in cand_tokens if t == nt])
    if hit:
        return hit

    # 4. last resort: "Real" is a club-form word too ("Real Valladolid CF" -> "Valladolid").
    #    Still the SAME token set and ambiguity-checked, so "Real Madrid"/"Real Sociedad" stay apart.
    nt2 = _tokens(name, keep_real=False)
    if nt2:
        hit = _only([c for c, t in cand_tokens if t and _tokens(c, keep_real=False) == nt2])
        if hit:
            return hit

    return None


def resolve_result(name: str, candidates: list[str]) -> str | None:
    """resolve() for RESULT settlement: the normal league-scoped tiers first; only if they find
    nothing, the API-Football result-spelling aliases (either direction). Predict-time matching
    must keep using resolve()."""
    hit = resolve(name, candidates)
    if hit or not name or not candidates:
        return hit
    n = norm(name)
    by_norm = {}
    for c in candidates:
        by_norm.setdefault(norm(c), c)
    return _alias_hit(n, _RESULT_ALIASES, by_norm)


def _spellings(v) -> tuple[str, ...]:
    return (v,) if isinstance(v, str) else tuple(v)


def _alias_hit(n: str, table: dict, by_norm: dict) -> str | None:
    """The ONE candidate an alias entry names for `n`, in either direction, else None.

    An entry is one club: its key and every spelling in its value. A value may be a tuple when the
    history holds the club under several spellings; if more than one of them is in the pool the
    pool itself carries a duplicate identity, and picking one half is refused (None)."""
    group = _alias_groups(id(table), table).get(n, frozenset())
    hits = {by_norm[m] for m in group if m != n and m in by_norm}
    return hits.pop() if len(hits) == 1 else None


_GROUP_CACHE: dict = {}


def _alias_groups(key, table: dict) -> dict:
    """{spelling: every spelling of that club}. Entries that share a spelling are one club, so
    the groups are closed transitively ('peterboro' ~ 'peterborough' ~ 'peterborough united')."""
    sig = (key, len(table), tuple(table.items()))
    hit = _GROUP_CACHE.get(key)
    if hit and hit[0] == sig:
        return hit[1]
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, v in table.items():
        for m in _spellings(v):
            parent[find(k)] = find(m)
    groups: dict = {}
    for x in list(parent):
        groups.setdefault(find(x), set()).add(x)
    out = {x: frozenset(groups[find(x)]) for x in parent}
    _GROUP_CACHE[key] = (sig, out)
    return out
