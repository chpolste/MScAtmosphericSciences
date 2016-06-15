import os, subprocess, json, datetime
from collections import namedtuple, OrderedDict

import numpy as np
import pandas as pd


__all__ = ["BUFRReader"]

__doc__ = """Parse BUFR files with the help of eccodes' bufr_dump."""


class BUFRReader:
    """BUFR reader based on eccodes' bufr_dump tool.
    
    Uses the flattened form of JSON output from bufr_dump and a crude parser
    to obtain Profile instances of the contents.

    Author's remark:
    The reader is targeted towards the type of BUFR I've needed to read, this
    is by no means a general implementation.
    """

    def __init__(self, bufr_dump="bufr_dump"):
        """Set up the reader, tell it how to find bufr_dump."""
        self._bufr_dump = bufr_dump
        assert "eccodes" in self._run_bufr_dump("-V")

    def _run_bufr_dump(self, arg):
        """Obtain the output of bufr_dump as text."""
        proc = subprocess.run([self._bufr_dump, "-jf", arg],
                stdout=subprocess.PIPE, universal_newlines=True)
        return proc.stdout

    def dump(self, path):
        """Obtain the output of bufr_dump as JSON loaded in Python."""
        return json.loads(self._run_bufr_dump(path))

    def read(self, path):
        """Obtain Profile instances for the content of a BUFR file."""
        dump = self.dump(path)
        metadata, *profiles = parser.parse(dump["messages"])
        out = []
        for columns, units in profiles:
            # Check for appropriate units
            assert "time" not in units or units["time"] == "s"
            assert "p" not in units or units["p"] == "Pa"
            assert "geopot" not in units or units["geopot"] == "gpm"
            assert "T" not in units or units["T"] == "K"
            assert "Td" not in units or units["Td"] == "K"
            assert "ff" not in units or units["ff"] == "m/s"
            assert "dd" not in units or units["dd"] == "deg"
            # Load data into dataframe, sort and apply simple quality control
            df = (pd.DataFrame.from_dict(columns, orient="columns", dtype=float)
                    .sort_values(by="time", ascending=True)
                    .dropna(how="any", axis="index")
                    .drop_duplicates(keep="first")
                    .reset_index(drop=True)
                    )
            if "time" in df: df["time"] = df["time"].astype(int)
            # Convert pressure to hPa
            if "p" in df: df["p"] = df["p"] / 100
        return df, metadata


class MessageStateParser:
    """A parser concept for the output of bufr_dump based on states.
    
    For each message check if it matches a state transition. If so, switch
    states, else stay in current state and continue to collect information.
    The register method allows to add new state transitions, parse applies
    the set up parser to a stream of messages.

    The output of the parse method is determined by the information collected
    in the states that were active during parsing. Each state can return a
    customized representation of the parsed contents or the Ellipsis singleton,
    which is removed in the total output.
    """

    def __init__(self, nostate=None):
        """Set up the parser.
        
        nostate is the state that is used at the beginning and when no other
        state is active (i.e. when a state decided to end outside of a
        transition rule.
        """
        self.transitions = []
        self._transitiontup = namedtuple("Transition", ["test", "state"])
        self.nostate = nostate if nostate is not None else Trash

    def register(self, test, state):
        """Add a state transition to the parser.

        Test should return True if a message triggers the transition to state.
        """
        assert callable(test)
        assert callable(state)
        self.transitions.append(self._transitiontup(test, state))
        return self # allow method chaining

    def parse(self, messages):
        """Parse a list of messages."""
        out = []
        state = self.nostate(None)
        for msg in messages:
            newstate = self.matchstate(msg)
            if newstate is None: # Stay in current state
                try:
                    state.put(msg)
                except AssertionError: # Current state wants to exit
                    out.append(state.get())
                    state = self.nostate(None)
            else: # Switch state
                out.append(state.get())
                state = newstate(msg)
        out.append(state.get())
        return [o for o in out if o is not ...]

    def matchstate(self, msg):
        """Return new state if transition applies, else None."""
        for test, state in self.transitions:
            if test(msg): return state
        return None


class Trash:
    """Subparser that throws everything away."""
    def __init__(self, msg): pass
    def put(self, msg): pass
    def get(self): return ...


class Header:
    """General Radiosonde Metadata"""

    select = {
            "cloudAmount": "cloud cover",
            "heightOfStationGroundAboveMeanSeaLevel": "altitude",
            "latitude": "latitude", "longitude": "longitude",
            "stationNumber": "station number",
            "year": "year", "month": "month", "day": "day",
            "hour": "hour", "minute": "minute", "second": "second"
            }

    def __init__(self, msg):
        self.content = {}

    def put(self, msg):
        key, value = msg["key"], msg["value"]
        # Discard field if not specified in self.select
        if key not in self.select.keys(): return
        # Rename field according to self.select
        key = self.select[key] # keep value
        self.content[key] = value

    def get(self):
        self.content["datetime"] = datetime.datetime(
                year=self.content["year"], month=self.content["month"],
                day=self.content["day"], hour=self.content["hour"],
                minute=self.content["minute"], second=self.content["second"],
                tzinfo=datetime.timezone.utc
                )
        self.content["timestamp"] = self.content["datetime"].timestamp()
        return self.content


class Sounding:
    """Extract colnames, units and data."""

    select = {
            "timePeriod": "time",
            "pressure": "p",
            "geopotentialHeight": "geopot",
            "airTemperature": "T",
            "dewpointTemperature": "Td",
            "windDirection": "dd",
            "windSpeed": "ff"
            }

    def __init__(self, msg):
        self.units = OrderedDict()
        self.columns = OrderedDict()
        self.columns_fixed = False
        if not msg["key"] == "extendedDelayedDescriptorReplicationFactor":
            # The low resolution profile is triggered by a NaN time value,
            # to obtain a full row of values, this value has to be put into
            # the collected data as well.
            self.put(msg)

    def put(self, msg):
        key, value, unit = msg["key"], msg["value"], msg["units"]
        # Discard field if not specified in self.select
        if key not in self.select.keys(): return
        # Rename field according to self.select
        key = self.select[key]
        if self.columns_fixed:
            assert key in self.columns.keys()
            self.columns[key].append(value)
        else:
            if key in self.columns:
                # Column has beed read before, from now on only fill new values
                # into existing columns, don't allow creation of new ones
                self.columns_fixed = True
                return self.put(msg)
            self.columns[key] = [value]
            self.units[key] = unit

    def get(self):
        assert len(set(len(col) for col in self.columns.values())) == 1
        return self.columns, self.units



# Assemble radiosonde profile parser:
# 1. section contains metadata (location, time, ...)
# 2. section is the profile
# 4. section contains instrument information (currently thrown away)

def _test_header(msg):
    return msg["key"] == "subsetNumber"

def _test_sounding(msg):
    return msg["key"] == "extendedDelayedDescriptorReplicationFactor"

def _test_footer(msg):
    return msg["key"] == "delayedDescriptorReplicationFactor"

parser = (
        MessageStateParser()
        .register(_test_header, Header)
        .register(_test_sounding, Sounding)
        .register(_test_footer, Trash)
        )
