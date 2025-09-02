CREATE TABLE IF NOT EXISTS einheiten_wind (
    -- Primary identifier
    einheit_mastr_nummer VARCHAR(50) NOT NULL PRIMARY KEY,
    
    -- Timestamps and dates
    datum_letzte_aktualisierung TIMESTAMP NOT NULL,
    netzbetreiberpruefung_datum DATE,
    registrierungsdatum DATE NOT NULL,
    inbetriebnahmedatum DATE,
    datum_endgueltige_stilllegung DATE,
    datum_beginn_voruebergehende_stilllegung DATE,
    datum_wiederaufnahme_betrieb DATE,
    geplantes_inbetriebnahmedatum DATE,
    datum_des_betreiberwechsels DATE,
    datum_registrierung_des_betreiberwechsels DATE,
    
    -- Location and address information
    lokation_mastr_nummer VARCHAR(50),
    land SMALLINT NOT NULL 
        CHECK (land IN (84, 90, 198, 66, 169, 106, 231, 206, 215, 269)),
    bundesland SMALLINT 
        CHECK (bundesland IN (1416, 1402, 1403, 1401, 1400, 1404, 1406, 1405, 1407, 1408, 1409, 1410, 1412, 1413, 1414, 1411, 1415)),
    landkreis VARCHAR(100),
    gemeinde VARCHAR(100),
    gemeindeschluessel VARCHAR(20),
    postleitzahl VARCHAR(10),
    gemarkung VARCHAR(100),
    flur_flurstuecknummern VARCHAR(200),
    strasse VARCHAR(200),
    strasse_nicht_gefunden SMALLINT CHECK (strasse_nicht_gefunden IN (0, 1)),
    hausnummer VARCHAR(20),
    hausnummer_nv SMALLINT CHECK (hausnummer_nv IN (0, 1)),
    hausnummer_nicht_gefunden SMALLINT CHECK (hausnummer_nicht_gefunden IN (0, 1)),
    adresszusatz VARCHAR(100),
    ort VARCHAR(100),
    laengengrad REAL,
    breitengrad REAL,
    
    -- Status information
    netzbetreiberpruefung_status SMALLINT NOT NULL 
        CHECK (netzbetreiberpruefung_status IN (2954, 2955, 3075)),
    einheit_systemstatus SMALLINT NOT NULL 
        CHECK (einheit_systemstatus IN (472, 473, 484, 490, 572)),
    einheit_betriebsstatus SMALLINT NOT NULL 
        CHECK (einheit_betriebsstatus IN (35, 37, 31, 38)),
    nicht_vorhanden_in_migrierten_einheiten SMALLINT NOT NULL 
        CHECK (nicht_vorhanden_in_migrierten_einheiten IN (0, 1)),
    
    -- Operator information
    anlagenbetreiber_mastr_nummer VARCHAR(50),
    alt_anlagenbetreiber_mastr_nummer VARCHAR(50),
    
    -- Plant identification
    name_stromerzeugungseinheit VARCHAR(200) NOT NULL,
    weic VARCHAR(50),
    weic_nv SMALLINT NOT NULL CHECK (weic_nv IN (0, 1)),
    weic_display_name VARCHAR(100),
    kraftwerksnummer VARCHAR(50),
    kraftwerksnummer_nv SMALLINT NOT NULL CHECK (kraftwerksnummer_nv IN (0, 1)),
    bestandsanlage_mastr_nummer VARCHAR(50),
    
    -- Technical specifications
    energietraeger SMALLINT NOT NULL 
        CHECK (energietraeger IN (2411, 2493, 2408, 2410, 2494, 2409, 2412, 2407, 2413, 2403, 2404, 2405, 2495, 2406, 2957, 2958, 2496, 2498, 2497)),
    bruttoleistung REAL NOT NULL,
    nettonennleistung REAL NOT NULL,
    anschluss_an_hoechst_oder_hoch_spannung SMALLINT CHECK (anschluss_an_hoechst_oder_hoch_spannung IN (0, 1)),
    einsatzverantwortlicher VARCHAR(200),
    fernsteuerbarkeit_nb SMALLINT CHECK (fernsteuerbarkeit_nb IN (0, 1)),
    fernsteuerbarkeit_dv SMALLINT CHECK (fernsteuerbarkeit_dv IN (0, 1)),
    einspeisungsart SMALLINT CHECK (einspeisungsart IN (688, 689)),
    gen_mastr_nummer VARCHAR(50),
    
    -- Wind-specific information
    name_windpark VARCHAR(200),
    wind_an_land_oder_auf_see SMALLINT
        CHECK (wind_an_land_oder_auf_see IN (888, 889)),
    seelage SMALLINT CHECK (seelage IN (640, 639)),
    cluster_nordsee SMALLINT 
        CHECK (cluster_nordsee IN (1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 2963)),
    cluster_ostsee SMALLINT 
        CHECK (cluster_ostsee IN (1540, 1541, 1542, 1543, 1544, 2970, 2962)),
    hersteller INTEGER,
    technologie SMALLINT,
    typenbezeichnung VARCHAR(200),
    nabenhoehe REAL,    
    rotordurchmesser REAL,
    
    -- Operational constraints and features
    rotorblattenteisungssystem SMALLINT CHECK (rotorblattenteisungssystem IN (0, 1)),
    auflage_abschaltung_leistungsbegrenzung SMALLINT CHECK (auflage_abschaltung_leistungsbegrenzung IN (0, 1)),
    auflagen_abschaltung_schallimmissionsschutz_nachts SMALLINT CHECK (auflagen_abschaltung_schallimmissionsschutz_nachts IN (0, 1)),
    auflagen_abschaltung_schallimmissionsschutz_tagsueber SMALLINT CHECK (auflagen_abschaltung_schallimmissionsschutz_tagsueber IN (0, 1)),
    auflagen_abschaltung_schattenwurf SMALLINT CHECK (auflagen_abschaltung_schattenwurf IN (0, 1)),
    auflagen_abschaltung_tierschutz SMALLINT CHECK (auflagen_abschaltung_tierschutz IN (0, 1)),
    auflagen_abschaltung_eiswurf SMALLINT CHECK (auflagen_abschaltung_eiswurf IN (0, 1)),
    auflagen_abschaltung_sonstige SMALLINT CHECK (auflagen_abschaltung_sonstige IN (0, 1)),
    nachtkennzeichnung SMALLINT CHECK (nachtkennzeichnung IN (0, 1)),
    buergerenergie SMALLINT CHECK (buergerenergie IN (0, 1)),
    
    -- Offshore specific
    wassertiefe REAL,
    kuestenentfernung REAL,
    
    -- Additional references
    eeg_mastr_nummer VARCHAR(50),
    
    -- Airborne wind technology
    technologie_flugwindenergieanlage SMALLINT 
        CHECK (technologie_flugwindenergieanlage IN (3163, 3164, 3165, 3166)),
    flughoehe REAL,
    flugradius REAL
);


-- Add comments to explain the enumeration values
COMMENT ON COLUMN einheiten_wind.land IS 'Country code (84=Germany, 90=Turkey, etc.)';
COMMENT ON COLUMN einheiten_wind.wind_an_land_oder_auf_see IS 'Wind location type: 888=Onshore, 889=Offshore';
COMMENT ON COLUMN einheiten_wind.technologie IS 'Wind technology code per MaStR catalog';
COMMENT ON COLUMN einheiten_wind.seelage IS 'Sea location: 639=North Sea, 640=Baltic Sea';
COMMENT ON TABLE einheiten_wind IS 'Wind energy units from German MaStR (Marktstammdatenregister)';