Usage

```
wget -e robots=off -r -np 'http://mag.gmu.edu/git-data/digital-rf/data/' 
ln -s mag.gmu.edu/git-data/digital-rf/data 
python drf.py
```

`drf.py` processes all data in W2NAF directory.

----

Why is read of KD1LE node 43-3 data so much slower?

`drf_plot.py` from `https://github.com/MITHaystack/digital_rf/tree/master/python/tools`

```
python drf_plot.py -i W2NAF/zip/OBS2024-05-30T00-00 -p power
```

gives

```
problem loading file W2NAF/zip/OBS2024-05-30T00-00
Traceback (most recent call last):
  File "/Users/weigel/Desktop/drf/drf_plot.py", line 1224, in <module>
    d = drf.read_vector(sstart, dlen, chans[chidx], subchan)
  File "/Users/weigel/miniconda3/lib/python3.9/site-packages/digital_rf/digital_rf_hdf5.py", line 1412, in read_vector
    z = self.read_vector_raw(start_sample, vector_length, channel_name, sub_channel)
  File "/Users/weigel/miniconda3/lib/python3.9/site-packages/digital_rf/digital_rf_hdf5.py", line 1474, in read_vector_raw
    raise IOError(estr % vector_length)
OSError: Number of samples requested must be greater than 0, not 0
```