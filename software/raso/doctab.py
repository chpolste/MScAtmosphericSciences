import zlib, io
import datetime as dt

import numpy as np
from doctab import DocTab, sql

from raso import Profile


__all__ = ["ProfileTab", "ProfileProcessor"]

__doc__ = """Store and retrieve Profile instances from a DocTab."""


def ProfileTab(connection, tablename, **kwargs):
    """A DocTab Instance with the raso_npz processor already set up."""
    doctab = DocTab(connection, tablename, **kwargs)
    ProfileProcessor.register_to(doctab)
    return doctab


class ProfileProcessor:
    """Save profiles to the database as compressed np.save output."""

    def input(self, doc, metadata, doctype):
        assert isinstance(doc, Profile)
        if doctype is None: doctype = self.default
        if doctype != "raso_npz":
            raise NotImplementedError("Unknown doctype '{}'".format(doctype))
        metadata = {} if metadata is None else metadata.copy()
        metadata.update(doc.metadata)
        if "datetime" in metadata: del metadata["datetime"]
        bytesio = io.BytesIO()
        np.save(bytesio, doc.data)
        data = sql.Blob(zlib.compress(bytesio.getvalue()))
        return data, metadata, doctype

    def output(self, doc, metadata, doctype, id):
        if doctype != "raso_npz":
            raise NotImplementedError("Unknown doctype '{}'".format(doctype))
        if "timestamp" in metadata:
            metadata["datetime"] = dt.fromutctimestamp(metadata["timestamp"])
        out = Profile(metadata=metadata)
        p.data = np.load(io.BytesIO(zlib.decompress(doc)))
        return out

    @classmethod
    def register_to(cls, table, **kwargs):
        processor = cls(**kwargs)
        for doctype in [None, "raso_npz"]:
            table.set_processor(inout=processor, doctype=doctype)

