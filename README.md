## Zprime-dilepton - A data analyis framework for the high-mass dilepton search based on NanoAOD

- Data analysis is performed based on the CMS NanoAOD data format using the [columnar approach](https://indico.cern.ch/event/759388/contributions/3306852/attachments/1816027/2968106/ncsmith-how2019-columnar.pdf), making use of the tools provided by [coffea](https://github.com/CoffeaTeam/coffea) package
- This framework uses the [copperhead](https://github.com/Run3HmmAnalysis/copperhead) framework developed for the H&rarr;µµ analysis by [Dmitry Kondratyev](https://github.com/kondratyevd) et. al. as a backend, re-implementing only analysis-specific code

## Framework structure, data formats, used packages
The input data for the framework should be in `NanoAOD` format.

The analysis workflow contains three stages:
- **NanoAOD processing** (WIP )includes event and object selection, application of corrections, and construction of new variables. All event weights, including normalization of MC events to luminosity, are applied already in this step. The data columns are handled via `coffea`'s `NanoEvents` format which relies on *jagged arrays* implemented in [Awkward Array](https://github.com/scikit-hep/awkward-1.0) package. After event selection, the jagged arrays are converted to flat [pandas](https://github.com/pandas-dev/pandas) dataframes and saved into [Apache Parquet](https://github.com/apache/parquet-format) files.
- **Analysis** (WIP) event categorization and production of histograms and unbinned datasets. In the future it might contain also the evaluation of MVA methods (boosted decision trees, deep neural networks),  The workflow is structured as follows:
  - Outputs of the NanoAOD processing (`Parquet` files) are loaded as partitions of a [Dask DataFrame](https://docs.dask.org/en/stable/dataframe.html) (similar to Pandas DF, but partitioned and "lazy").
  - The Dask DataFrame is (optionally) re-partitioned to decrease number of partitions.
  - The partitions are processed in parallel; for each partition, the following sequence is executed:
    - Partition of the Dask DataFrame is "computed" (converted to a Pandas Dataframe).
    - Definition of event categories.
    - Creating 1D or 2D histograms using [scikit-hep/hist](https://github.com/scikit-hep/hist).
    - Saving histograms.
    - (Optionally) Saving individual columns (can be used later for unbinned fits).

- **Result** (WIP) contains / will contain plotting, parametric fits, preparation of datacards for statistical analysis. The plotting is done via [scikit-hep/mplhep](https://github.com/scikit-hep/mplhep).

## Job parallelization
The analysis workflow is efficiently parallelised using [dask/distributed](https://github.com/dask/distributed) with either a local cluster (uses CPUs on the same node where the job is launched), or a distributed `Slurm` cluster initialized over multiple computing nodes. The instructions for the Dask client initialization in both modes can be found [here](docs/dask_client.md).

It is possible to create a cluster with other batch submission systems (`HTCondor`, `PBS`, etc., see full list in [Dask-Jobqueue API](https://jobqueue.dask.org/en/latest/api.html#)) and support will be added in collaboration with users.

## Installation instructions
Work from a `conda` environment to avoid version conflicts:
```bash
module load anaconda/5.3.1-py37
conda create --name hmumu python=3.7
source activate hmumu
```
Installation:
```bash
git clone -recursive https://github.com/minxiyang/Zprime-Dilepton
cd Zprime-Dilepton
python3 -m pip install --user --upgrade -r requirements.txt
```
If accessing datasets via `xRootD` will be needed:
```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
. setup_proxy.sh
```

after the first setup, the correct environment can be set using
```bash
source setup.sh
```

alternatively, for use without a grid proxy, use:
```bash
source setupWithoutProxy.sh
```
## Where to find denifitions

`config/parameters.py` contains defintions about object selections, triggers, JSON files, etc
`datasets_muon.py` and `datasets_electron.py` contain the lists of NanoAOD datasets used for the analysis
`config/cross_sections.py` contains the list of cross sections for MC processes
`config/variables.py` contains the list of variables available for plotting with the information about labels and plot ranges

## Usage instructions

For the first two steps (NanoAOD processing and Analysis) it is recommended to use a Dask cluster with at least 100 workers (1 CPU core per worker) to keep processing times down. The last step can be done locally. The three steps can be performed using the base commands
```bash
python run_nanoProcessing.py
python run_analysis.py
python run_results.py
```
To interact with the Dask cluster, the variable `node_ip` in these scripts has to be set to the IP address of the cluster. To trigger submission of the processing to the Dask cluster, the ``--sl`` argument is used to pass the port number under which the cluster can be accessed. Otherwise, a local Dask cluster will be initiated and the processing will be performed on the local machine. This is the preferred option for small jobs and for debug purposes. 

The code executed by the `run_nanoProcessing.py` script is located in the folder `processNano`. The core of each processing are the `xxx_processor.py` scripts, which contain the code to process the event information and transform them into the dataframes with the requested information. Helper code and implementations for corrections and uncertainties are imported inside the processors. 

The code exected by the `run_analysis.py` script is located in the folder `doAnalysis`. Currently, alll the relevant code is located within `doAnalysis/postprocessor.py` and `doAnalysis/histogrammer.py`. The splitting into different b-jet multiplicities is done in `doAnalysis/categorizer.py` Inside `run_analysis.py` the datasets to be processed and the quantities for with 1D, 2D and unbinned output are to be created. 

The code executed by the `run_results.py` script is located in the folder `produceResults`. The only part of the code currently used in `produceResults/plotter.py`. Which plots to create, which processes to include in them and how to group MC processes into stacked histograms is configured in the `run_result.py` script.


## Credits
- **Zprime-dilepton** [Minxi Yang](https://github.com/minxiyang), [Jan-Frederik Schulte](https://github.com/JanFSchulte)
- **Copperhead backend:** [Dmitry Kondratyev](https://github.com/kondratyevd), [Arnab Purohit](https://github.com/ArnabPurohit), [Stefan Piperov](https://github.com/piperov)


