import os
from tempfile import TemporaryDirectory
from subprocess import Popen

from monortm import paths


def chunks(xs, n):
    """Split xs into n chunks as even as possible."""
    l = len(xs)
    # Split up divisible part
    counts = [l//n]*n
    # Distribute remaining elements
    for i in range(l % n):
        counts[i] += 1
    print(counts)
    current = 0
    for count in counts:
        yield xs[current:current+count]
        current += count


def run(config, profiles, processes=0):
    """"""
    if processes <= 0:
        processes = os.cpu_count()
    

class MonoRTM:
    """"""

    def __init__(self, config, profile):
        self.config = config
        self.profile = profile
        self._folder = None
        self._process = None

    def run(self, async=False):
        """Run MonoRTM with the given configuration and profile.
        
        If async == True, MonoRTM will run in the background. Use wait, poll
        and cleanup to control the process manually. Output can be obtained
        from the output property of the instance after MonoRTM finished
        execution.
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
            self._process = Popen([paths["MonoRTM"]], cwd=self._folder.name)
        except:
            self.cleanup()
            raise
        if not async:
            self.wait()
            self.cleanup()

    @property
    def output(self):
        return self._output

    def wait(self):
        return None if self._process is None else self._process.wait()

    def poll(self):
        return None if self._process is None else self._process.poll()

    def cleanup(self):
        if self.poll() is not None:
            self._process.kill()
            self._process = None
        if self._folder is not None:
            self._folder.cleanup()
            self._folder = None

    def __del__(self):
        self.cleanup()

