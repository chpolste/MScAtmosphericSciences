"""Parse BUFR files with the help of ecCodes' bufr_dump.

Specialized on radiosounding data. Calls bufr_dump -jf on a bufr file, reads
the JSON output and uses a crude state machine to convert the contents into
a pandas dataframe and a dictionary containing additional metadata.

The performance bottleneck is bufr_dump, which for some reason takes quite
a long time to write JSON to stdout...

ecCodes: https://software.ecmwf.int/wiki/display/ECC/ecCodes+Home
         Version 0.13.0 was used during development.
"""


import os, subprocess, json, datetime
from collections import namedtuple, OrderedDict

import numpy as np
import pandas as pd


__all__ = ["BUFRReader"]


class BUFRReader:
    """Read BUFR files containing radiosonde data.

    Calls ecCodes' bufr_dump with subprocess.run, reads the output with json
    and then rearranges the contents into dataframes and a dictionary
    containing metadata.

    Supported fields: (see also Sounding class)

    name                    renamed to
    ----------------------------------
    timePeriod              time
    pressure                p
    geopotentialHeight      z
    airTemperature          T
    dewpointTemperature     Td
    windDirection           dd
    windSpeed               ff

    Pressure is converted from Pa to hPa.
    """

    def __init__(self, bufr_dump="bufr_dump"):
        """Set up the reader, tell it how to call bufr_dump."""
        self._bufr_dump = bufr_dump
        assert "eccodes" in self._run_bufr_dump("-V")

    def _run_bufr_dump(self, arg):
        """Obtain the output of bufr_dump as text."""
        # -jf enables flattened JSON output
        proc = subprocess.run([self._bufr_dump, "-jf", arg],
                stdout=subprocess.PIPE, universal_newlines=True)
        return proc.stdout

    def dump(self, path):
        """Obtain the output of bufr_dump as JSON loaded into Python."""
        return json.loads(self._run_bufr_dump(path))

    def read(self, path):
        """Read the given BUFR file and return dataframes and a metadata dict."""
        dump = self.dump(path)
        metadata, *profiles = parser.parse(dump["messages"])
        out = []
        for columns, units in profiles:
            # Check for appropriate units
            assert "time" not in units or units["time"] == "s"
            assert "p" not in units or units["p"] == "Pa"
            assert "z" not in units or units["z"] == "gpm"
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
            out.append(df)
        return out, metadata


class MessageStateParser:
    """A parser concept for the output of bufr_dump based on states.

    For each message check if it matches a state transition. If so, switch
    states else stay in current state and continue to collect information.
    The register method allows to add new state transitions, parse applies
    the set up parser to a stream of messages. State transitions are
    independent of the current state so this is not a typical state machine.

    The output of the parse method is determined by the information collected
    in the states that were active during parsing. Each state can return a
    customized representation of the parsed contents or the Ellipsis singleton,
    which is removed in the total output.
    """

    def __init__(self, nostate=None):
        """Set up the parser.

        nostate is the state that is used at the beginning and when no other
        state is active (i.e. when a state decided to end itself without the
        application of a transition rule.
        """
        self.transitions = []
        self._transitiontup = namedtuple("Transition", ["test", "state"])
        self.nostate = nostate if nostate is not None else Trash

    def register(self, test, state):
        """Add a state transition to the parser.

        Test should return True if a message triggers the transition to state.
        Care has to be taken in the ordering of state registration as state
        registered earlier have priority when multiple transitions apply.

        state is called during a state transition with the message that
        triggered the state transition (the state's put method is not called
        with the transition message after a transition so be sure to handle
        anything related to that message inside the state's initialization). It
        should have a put method that accepts new messages and a get method
        that returns the collected information when the state exits. If a state
        wants to exit an AssertionError can be raised during put.
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
        # Remove Ellipsis singletons from output
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
    """General radiosonde metadata.

    Collects every key-value pair specifies in the select attribute and adds
    datetime and timestamp attributes.
    """

    select = {
            "cloudAmount": "cloud cover",
            "heightOfStationGroundAboveMeanSeaLevel": "altitude",
            "latitude": "latitude", "longitude": "longitude",
            "stationNumber": "station number",
            "year": "year", "month": "month", "day": "day",
            "hour": "hour", "minute": "minute", "second": "second"
            }

    def __init__(self, msg):
        # The initial message contains
        self.content = {}

    def put(self, msg):
        key, value = msg["key"], msg["value"]
        if key not in self.select.keys(): return
        # Rename field
        key = self.select[key]
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
    """Extract colnames, units and data and arranges them in a dataframe."""

    select = {
            "timePeriod": "time",
            "pressure": "p",
            "geopotentialHeight": "z",
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

