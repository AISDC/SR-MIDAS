# SR-MIDAS
### Super resolution workflow for MIDAS reconstruction

<hr />

## OPTION 1: AUTO-INSTALLATION: RECOMMENDED

Download only the 'super_res_install.py' file to your local system. In future, it will be automatically included within the 'MIDAS/utils' directory.

Start a terminal within the MIDAS directory (it is critical to work within the MIDAS directory). This is the same directory where MIDAS was cloned.

Run the 'super_res_install.py' file from here as shown below:

'''bash
python ../../super_res_install.py
'''

This will automatically install/update the super-resolution workflow.

## ## OPTION 2: MANUAL INSTALLATION: NOT RECOMMENDED

This implementation requires manually copying the files included here into an existing MIDAS installation. The target directory is as below:

```bash
MIDAS/FF_HEDM/v7/
```

The data to be copied is as below:

- **trained_mods_CNNSR**: dir: This contains trained models for inference.
- **ff_MIDAS_sr.py**: file: This is alternative implementation of 'ff_MIDAS.py' that supports SR workflow.
- **super_res_process.py**: file: Python file that runs the SR workflow and is called from within the 'ff_MIDAS_sr.py'.
- **sr_config.json**: file: Configuration file to control behaviour of SR workflow.


## Running MIDAS with super-resolution

```bash
python ~/MIDAS/FF_HEDM/v7/ff_MIDAS_sr.py \
    -paramFN ps.txt \
    -fileName data.edf.ge5 \
    -nCPUs 40 \
    -numFrameChunks 100 \
    -preProcThresh 30 \
    -doPeakSearch 0 \
    -runSR 1 \
    -srfac 8
```

NOTE: In the above execution, take a note of the following:

1) 'ff_MIDAS_sr.py' was called and not 'ff_MIDAS.py'.

2) -doPeakSearch 0 : MIDAS peak search is disabled since this task will transfer to SR workflow.

3) -runSR 1: Triggers SR workflow

4) -srfac 8: Defining the super-resolution factor (possible values: 2, 4 or 8)


## SRlogs
In the result directory, a new 'SRlogs' directory is createed that contains the logging for SR workflow.
