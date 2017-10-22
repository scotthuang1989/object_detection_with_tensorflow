from six.moves import urllib
import os
import sys


def maybe_download(dest_directory, filename, data_url):
  """Download and extract the tarball from Alex's website."""
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)

  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
