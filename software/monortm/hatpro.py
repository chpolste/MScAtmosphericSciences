from monortm.records import Record12, Record13, Record131, Record132, Record14

ghz2icm = 1 / 29.9792458 # = 1.0e9 / speed of light in cm/s

config = [
        Record11(TOKEN="$", CXID="created for HATPRO simulation"),
        Record12(IHIRAC=1, ICTNM=1, IEMIT=1, IPLOT=1, IATM=0, IXSECT=0, IBRD=1),
        Record13(V1=-99., V2=-99., DVSET=None, NMOL_SCAL=None),
        Record131(NWN=14),
        Record132(WN=22240*MHz2icm),
        Record132(WN=23040*MHz2icm),
        Record132(WN=23840*MHz2icm),
        Record132(WN=25440*MHz2icm),
        Record132(WN=26240*MHz2icm),
        Record132(WN=27840*MHz2icm),
        Record132(WN=31400*MHz2icm),
        Record132(WN=51260*MHz2icm),
        Record132(WN=52280*MHz2icm),
        Record132(WN=53860*MHz2icm),
        Record132(WN=54940*MHz2icm),
        Record132(WN=56660*MHz2icm),
        Record132(WN=57300*MHz2icm),
        Record132(WN=58000*MHz2icm),
        Record11(TOKEN="%")
        ]
