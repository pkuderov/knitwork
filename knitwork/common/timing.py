from timeit import default_timer

from knitwork.common.base import print_with_timestamp, to_readable_num


timer = default_timer

class Timer:
    def __init__(self):
        self.t_start = timer()
        self.last_elapsed = 0.0
        self.n_last_iters = 0
        self.total_elapsed = 0.0
        self.n_iters = 0

    def new(self):
        self.t_start = timer()

    @property
    def elapsed(self):
        return timer() - self.t_start

    def commit(self, n_iters=1, new_after=False):
        dt = self.elapsed
        self.last_elapsed = dt
        self.total_elapsed += dt
        self.n_last_iters = n_iters
        self.n_iters += n_iters

        if new_after:
            self.new()
        return dt

    def avg(self, n_iters=None, last=False):
        if n_iters is None:
            n_iters = self.n_iters if not last else self.n_last_iters
        n_iters = max(1, n_iters)

        elapsed = self.total_elapsed if not last else self.last_elapsed
        return elapsed / n_iters

    def fps(self, n_iters=None, last=False):
        avg = self.avg(n_iters, last)
        avg = max(1e-10, avg)
        return 1.0 / avg

    def print_elapsed(self, tag='', last=False):
        """Print total or last elapsed time."""
        t = self.total_elapsed if not last else self.last_elapsed

        t, dt_sx = to_readable_num(t)
        typ = "dT" if last else "TT"
        print(f'"{tag}" {typ}: {t:.2f}{dt_sx}')

    def print_avg(self, tag='', n_iters=None, last=False):
        """Print average iter time"""
        avg = self.avg(n_iters, last)
        avg, avg_sx = to_readable_num(avg)
        lst_sx = " (last)" if last else ""
        print(f'"{tag}" iT{lst_sx}: {avg:.2f}{avg_sx}')
    
    def print_fps(self, tag='', n_iters=None, last=False):
        """Print average FPS"""
        fps = self.fps(n_iters, last)
        fps, fps_sx = to_readable_num(fps)
        lst_sx = " (last)" if last else ""
        print(f'"{tag}" FPS{lst_sx}: {fps:.2f}{fps_sx}')

    def print(self, *args):
        """Regular print with the '[<elapsed seconds>]' prefix. """
        assert self.n_iters == 0, 'To avoid mistakes the method forbids committing'
        print_with_timestamp(self.elapsed, *args)
