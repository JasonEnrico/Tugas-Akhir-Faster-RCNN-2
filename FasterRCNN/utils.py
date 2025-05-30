import torch as t

class CSVLog:
  """
  Logs to a CSV file.
  """
  def __init__(self, filename):
    self._filename = filename
    self._header_written = False

  def log(self, items):
    keys = list(items.keys())
    file_mode = "a" if self._header_written else "w"
    with open(self._filename, file_mode) as fp:
      if not self._header_written:
        fp.write(",".join(keys) + "\n")
        self._header_written = True
      values = [ str(value) for (key, value) in items.items() ]
      fp.write(",".join(values) + "\n")
