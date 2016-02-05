import sys
import time
from datetime import timedelta


class ProgressBar:
    """
    Instantiate a progress bar. It assumes values from 0 to end_value - 1.

    Usage:
    >>> bar = ProgressBar(end_value=100, text="Iteration")
    >>> bar.start()
    >>> for i in range(0, 100):
    >>>    time.sleep(1)
    >>>    bar.update(i)
    """

    def __init__(self, end_value=100, text=None, count=False, bar_length=20):
        """
        Initialize the progress bar.

        :param end_value:   Maximum number of iterations
        :param text:        Text to be displayed before the bar
        :param count:       Whether to display the count of iterations
        :param bar_length:  Number of hashes to display
        """
        self.end_value = end_value
        self.current = 0
        self.bar_length = bar_length
        self.start_time = 0
        self.count = count
        if text:
            self.text = text
        else:
            self.text = "Progress"

    def _reset(self):
        """
        Reset the bar.
        """
        self.current = 0
        self.start_time = 0

    def start(self):
        """
        Start the timer and displaying the bar.
        """
        self._reset()
        # Initialize starting time
        self.start_time = int(time.time())
        # Start displaying the bar
        if self.count:
            sys.stdout.write("\r{0} ({4:>{5}}/{3}): [{1}] {2}%".format(self.text,
                                                                       ' ' * self.bar_length,
                                                                       0,
                                                                       self.end_value,
                                                                       0,
                                                                       len(str(self.end_value))))
        else:
            sys.stdout.write("\r{0}: [{1}] {2}%".format(self.text,
                                                        ' ' * self.bar_length,
                                                        self.current))
        sys.stdout.flush()

    def update(self, new_val):
        """
        Update the bar with new_val.

        :param new_val: New current value
        """
        self.current = new_val + 1
        if self.current <= self.end_value:
            percent = float(self.current) / self.end_value
            hashes = '#' * int(round(percent * self.bar_length))
            spaces = ' ' * (self.bar_length - len(hashes))
            if self.count:
                sys.stdout.write("\r{0} ({4:>{5}}/{3}): [{1}] {2}%".format(self.text,
                                                                           hashes + spaces,
                                                                           int(round(percent * 100)),
                                                                           self.end_value,
                                                                           self.current,
                                                                           len(str(self.end_value))))
            else:
                sys.stdout.write("\r{0}: [{1}] {2}%".format(self.text,
                                                            hashes + spaces,
                                                            int(round(percent * 100))))
            sys.stdout.flush()
            if self.current == self.end_value:
                elapsed_time = timedelta(seconds=(int(time.time()) - self.start_time))
                sys.stdout.write(" Elapsed time: %s\n" % elapsed_time)


def progress_bar_test(end_value, text, bar_length, count):
    bar = ProgressBar(end_value=end_value, text=text, count=count, bar_length=bar_length)
    bar.start()
    for i in range(0, end_value):
        time.sleep(1)
        bar.update(i)


if __name__ == '__main__':
    progress_bar_test(100, text='Iterations', count=True, bar_length=20)
