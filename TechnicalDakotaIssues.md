## Technical issues
These specifically concern the python wheel (itis-dakota);
they may or may not arise when using the dakota executable or when using dakota
as a lib. These have been observed and mostly resolved through time.

1) variables and functions names are case insensitive, and some of the Ascii
non alphanumeric chars are non valid.
Suggestion : have a systematic name conversion internally into x1, x2, x3,...
For input variables, and y1, y2,... For output var. Make this invisible to the
user, systematic for Dakota. This solves all pb and allows you to easily
manipulate/search variables internally: for ex when gathering data from
files/stdout which only output values in a specific order, you can easily
reassign them to xi or yi. Whenever the user/osparc interact with variables the
conversion will take place, they never see these internal names, only those
they choose. Very clean. 

2) Seeds must be non zero, positive, floats are accepted but shouldn't be used imo. 
Floats are bad due to rounding errors (rounding may vary with hardware) and
offer no advantage over integers. I suggest to build a map into strictly
positive unsigned 64 bits ints. My guess is that Dakota just static casts the
values into a uint64 on the C++ side anyway.

3) Isolation issues: When calling dakota methods successively I've ran into the
pb where some sort of state remains between calls. In some instances data may
even be shared. Isolating dakota calls into distinct processes solves the pb
(distinct threads are not enough!).

4) The many Dakota methods are inconsistent regarding whether they support
batch or non-batch modes. Some methods even support batch everywhere except
during the last iteration. The best way seems to let a wrapping function detect
the mode requested by Dakota at runtime at every call/iteration; this can be
done by simply detecting whether a list or dict is passed in by Dakota. 

5) Dakota generated output on stdout can overflow the OS stdout buffer. We need
a special wrapper to direct very large textual output into files. Most python
packages I found are not able to handle such output sizes properly, I built a
little py file to do this right. 

6) Ill conditioned matrices happen occasinally when correlation matrices are
involved (GP modeling, optimization, etc). These cannot be totally avoided,
they depend on user input. They must be handled at run time.

7) Multi objective optimization does not always output solutions that are
strictly pareto. Some non-dominant values are kept (prob to smooth out the
spread on the pareto surface). This can become an issue when relying on the
pareto property (hypervolume metric, etc).

8) Tabular data file are sometimes inconsistent across methods. Don't let the
user directly interact/modify them. I think it's more robust to handle all data
on our side and present the results to the in a consistent way.

9) Parsing for data in stdout seems quite reliable, but the format is not
consistent across methods and versions.

10) Handling hd5 data files might be dangerous if you need to run multiple
Dakota calls or methods. I've managed to avoid them so far.

11) Manual prescaling is very often necessary. My suggestion is to normalize all
input variables between 0 and 1 or using log (offering both options will be
useful), without the user/osparc noticing similarly to the variable naming
conversion case. 

12) A manual base surrogate may be useful if the user only wants to play with
finite datasets. A Dakota workflow should offer both possibilities: use a full
function/study as base, or build a base surrogate based on user (finite) data. 

13) Effectiveness of optimization highly depends on the Dakota parametrization.
The unrecommended "elitism" keyword seems to help spreading the data along the
pareto.

14) GP modeling should offer a nugget option. Important for noisy input. 

15) Model validation is very incomplete. Dakota only offers to compute few
basic metrics (MAE, RMSE, etc) without disclosing the residuals.

16) Inverse uncertainty quantification (calibration) does not seem to work
well: the computed posteriors may lack precision for some practical
applications. 

17) Surrogate model serialization only works under very specific conditions.
For GP alone, various dakota packages may handle this very differently and it
is recomended to use the same dakota configuration files to save and reload
models. A more generic and robust approach is to simply retrain surrogates on
the fly based on saved training data sets, and never assume a reloaded model
will be exactly the same.   

18) Dakota GP modeling essentially uses an elaborate version of simple kriging:
no variograms involved, process means are estimated outside of the kriging
equations. This may imply inaccuracies in the presence of complicated types of
anisotropies. Appropriately transforming the input space a priory may improve
surrogate accuracy. In the global case we cannot expect dakota to always create
good models. This issue is mostly inexistent for local GP surrogates (for many
of the surrogate based methods). 