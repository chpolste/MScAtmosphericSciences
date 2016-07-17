"""Writing input files, running MonoRTM and parsing output files."""

import os
from tempfile import TemporaryDirectory
from subprocess import Popen

from monortm import paths


def write(f, records):
    """Write records to a file."""
    for rec in records:
        f.write(str(rec))
        f.write("\n")


class MonoRTM:
    """MonoRTM wrapper."""

    def __init__(self, config, profile):
        self.config = config
        self.profile = profile
        self._folder = None
        self._process = None
        self.raw_output = None

    def run(self, async=False):
        """Run MonoRTM with the given configuration and profile.

        If async == True, MonoRTM will run in the background. Use wait, poll
        and cleanup to control the process manually. Output is not
        automatically fetched after execution, it has to be fetched manually
        before cleanup by calling any of the output properties or
        fetch_output. Then it is also available after cleanup.
        """
        self._folder = TemporaryDirectory(prefix="MonoRTM_")
        monortm_in = os.path.join(self._folder.name, "MONORTM.IN")
        monortm_prof = os.path.join(self._folder.name, "MONORTM_PROF.IN")
        tape3 = os.path.join(self._folder.name, "TAPE3")
        try:
            with open(monortm_in, "w") as f:
                write(f, self.config)
            with open(monortm_prof, "w") as f:
                write(f, self.profile)
            os.symlink(paths["TAPE3"], tape3)
            self._process = Popen([paths["MonoRTM"], "2>&1 /dev/null"],
                    cwd=self._folder.name)
            if not async:
                self.wait()
                self.fetch_output()
                self.cleanup()
        except:
            self.cleanup()
            raise

    def fetch_output(self):
        monortm_out = os.path.join(self._folder.name, "MONORTM.OUT")
        if not os.path.exists(monortm_out):
            raise IOError("MONORTM.OUT not available.")
        with open(monortm_out, "r") as f:
            self.raw_output = f.read()

    @property
    def output(self):
        """Raw output split into chunks corresponding to profiles."""
        if self.raw_output is None:
            self.fetch_output()
        sep = "MONORTM RESULTS"
        return [sep + block for block in self.raw_output.split(sep) if block]

    @property
    def brightness_temperatures(self):
        """Brightness temperatures."""
        return [[float(line[15:26]) for line in block.splitlines()[4:]]
                for block in self.output]

    def wait(self):
        return None if self._process is None else self._process.wait()

    def poll(self):
        return None if self._process is None else self._process.poll()

    def cleanup(self):
        """Remove folder and kill process if still running."""
        if self._process is not None:
            if self.poll() is None:
                self._process.kill()
            self._process = None
        if self._folder is not None:
            self._folder.cleanup()
            self._folder = None

    def __del__(self):
        self.cleanup()

    @classmethod
    def run_distributed(cls, config, profiles, get="output", processes=0):
        """Run multiple profiles. Can distribute work on multiple processes."""
        assert hasattr(cls, get)
        if processes <= 0:
            processes = os.cpu_count()
        models = []
        output = []
        for chunk in chunks(profiles, processes):
            if not chunk: continue
            combined_profile = [rec for profile in chunk for rec in profile]
            model = cls(config, combined_profile)
            model.run(async=True)
            models.append(model)
        for model in models:
            model.wait()
            output.extend(getattr(model, get))
            model.cleanup()
        return output


def chunks(xs, n):
    """Split xs into n chunks as even as possible."""
    l = len(xs)
    # Split up divisible part
    counts = [l//n]*n
    # Distribute remaining elements
    for i in range(l % n):
        counts[i] += 1
    current = 0
    for count in counts:
        yield xs[current:current+count]
        current += count

