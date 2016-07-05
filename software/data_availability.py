import datetime as dt
import pandas as pd
from dbtoolbox import DatabaseToolbox

db = DatabaseToolbox("../data/amalg.db")

cosmos = db.as_dataframe("""
        SELECT valid FROM profiles WHERE kind="cosmo7"
        """)["valid"].map(pd.Timestamp.utcfromtimestamp)

rasos = db.as_dataframe("""
        SELECT valid FROM profiles WHERE kind="raso"
        """)["valid"].map(pd.Timestamp.utcfromtimestamp)

nordkettes = db.as_dataframe("""
        SELECT valid FROM nordkette
        """)["valid"].map(pd.Timestamp.utcfromtimestamp)
        
hatpros = db.as_dataframe("""
        SELECT valid FROM hatpro WHERE kind="hatpro_blb" OR kind="hatpro_brt"
        """)["valid"].map(pd.Timestamp.utcfromtimestamp)

out = ["Data Availability\n\nN = Nordkette\nR = Radiosounding\nC = COSMO7 Sounding\nH = HATPRO"]
last_month = 12
last_year = 1998
start = dt.datetime(1999, 1, 1, 0, 0)
while start < dt.datetime(2016, 5, 1, 0, 0):
    if last_year != start.year:
        last_year = start.year
        out.append("\n\n===== {} =====\n".format(last_year))
        out.append("     " + " ".join("{:>4}".format(i) for i in range(1, 32)))
        print(last_year)
    if last_month != start.month:
        out.append("\n")
        last_month = start.month
        out.append("{:>2} | ".format(last_month))
    end = start + dt.timedelta(days=1)
    has = lambda x: ((start <= x) & (x < end)).any()
    cosmo = "C" if has(cosmos) else "_"
    raso = "R" if has(rasos) else "_"
    nordkette = "N" if has(nordkettes) else "_"
    hatpro = "H" if has(hatpros) else "_"
    out.append(nordkette + raso + cosmo + hatpro + " ")
    start = end

with open("../misc/data_availability.txt", "w") as f:
    f.write("".join(out))
