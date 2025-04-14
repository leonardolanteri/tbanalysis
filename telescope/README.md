# Telescope analysis

## Format of data

Input data are json files, mainly read from here:
```
```

Processed as awkward arrays with several fields, example of an event:

```
{'event': 1331,
 'l1counter': 72,
 'row': [3, 3, 1],
 'col': [12, 12, 12],
 'tot_code': [83, 81, 137],
 'toa_code': [56, 537, 564],
 'cal_code': [158, 167, 168],
 'elink': [0, 0, 0],
 'chipid': [148, 144, 152],
 'bcid': 1391,
 'nhits': [1, 1, 1]}
```

## How to run

Analysis should be run by selecting one pixel-triplet (one pixel per layer) by hand:
```
python3 run_bootstrap.py --ipix 1 12 --jpix 3 12 --kpix 3 12
```
You can additionally require that the layers only have that exact pixel, but it is not mandatory

## To do's

### Implement track reconstruction to automatically select pixel-triplets

Actions, work on ```utils/TrackReconstructor.py```:

1. Work on deriving some aligment to correct the hits position before tracks are reconstructed.
2. Reconstruct tracks to identify pixels that are likely to come from the beam in the three layers.
3. Final format for the pixel triplers should be read by ```run_bootstrap.py```, so output events would ideally be a cleaned version on input events, only containing pixels matched to tracks.
