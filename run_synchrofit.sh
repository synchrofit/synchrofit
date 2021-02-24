#! /usr/bin/bash

usage()
{
echo "run_synchrofit.sh [-o option]
  -o option : Select where to source input spectrum." 1>&2;
exit 1;
}

while getopts ':o:' OPTION
do
    case "$OPTION" in
    o)
        option=${OPTARG} ;;
    ? | : | h)
            usage ;;
  esac
done

if [[ -z ${option} ]]
then
    usage
fi

# set execution options here
workdir="/home/sputnik/Documents/software/synchrofit/"
datafile="test_spectra.dat"
freq_unit='MHz'
flux_unit='Jy'
fit_type='JP'
nfreqplots=100
mcLength=100
sigma_level=2

if [[ "$option" == "file" ]]
then
    python3 synchrofit.py \
        --workdir $workdir \
        --data $datafile \
        --freq_unit $freq_unit\
        --flux_unit $flux_unit \
        --fit_type $fit_type \
        --nfreqplots $nfreqplots \
        --mcLength $mcLength \
        --sigma_level $sigma_level \
        --plot \
        --write_model
else
    python3 synchrofit.py \
        --workdir $workdir \
        --freq 122 151 189 215 399 887 1400 2000 2868 5500 9000 \
        --flux 0.260860 0.214700 0.181840 0.170609 0.108001 0.064071 0.042400 0.028180 0.018669 0.008059 0.004102 \
        --err_flux 0.017 0.015 0.001 0.00888 0.011 0.0032 0.00305 0.0021 0.00116 0.00055 0.00029 \
        --freq_unit $freq_unit\
        --flux_unit $flux_unit \
        --fit_type $fit_type \
        --nfreqplots $nfreqplots \
        --mcLength $mcLength \
        --sigma_level $sigma_level \
        --plot \
        --write_model
fi