"""
config.py
Central configuration for the MR-MOU-D-Tree framework.

Uncertainty Aware Multiobjective Metamodeling for Robust Selection of
Measles and Rubella Immunization Interventions in Bangladesh.

DATA PROVENANCE
---------------
This version replaces the previous fully synthetic division modifiers with
REAL, publicly reported Bangladesh data:

  * Division populations         ->  BBS Population and Housing Census 2022
                                     (PEC adjusted division totals).
  * National under-5 birth cohort->  BBS Census 2022 age structure and
                                     UN World Population Prospects 2024.
  * Division urban share         ->  BBS Census 2022 urban/rural split.
  * National MR1 / MR2 coverage  ->  Bangladesh Coverage Evaluation Survey
                                     (CES 2023): MR1 ~ 0.86, MR2 ~ 0.807, and
                                     WHO/UNICEF WUENIC.
  * 2026 outbreak division burden->  WHO Disease Outbreak News DON598
                                     (4 April 2026) and WHO SEARO situation
                                     reports (division incidence per million).

Coverage scenarios and the stochastic simulation replications remain synthetic
so the pipeline can be released openly, but every population, cohort and burden
anchor below is a published figure. All random draws are seeded so every
figure, table and number is reproducible.
"""

import numpy as np

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
GLOBAL_SEED = 20260707
RNG = np.random.default_rng(GLOBAL_SEED)

# ----------------------------------------------------------------------
# Experimental design sizes
# ----------------------------------------------------------------------
N_SAMPLE_VECTORS = 10000
CLUSTER_GRID = [100, 200, 400, 600, 800, 1000]
N_CENTROIDS = 400
N_DRAWS = 60
HORIZON_WEEKS = 260
M_POPULATIONS = 1000
N_ENSEMBLE = 25
CVAR_ALPHA = 0.90
DELTA_ROBUST = 0.20
DIST_THRESHOLD = 0.025

# Held-out and out-of-distribution test design (rigorous train/test)
TEST_FRACTION = 0.25
CALIB_FRACTION = 0.20
OOD_MR1_CUTOFF = 0.92
SPLIT_SEED = 424242

# Bootstrap / Monte-Carlo sizes (scaled down in QUICK mode below)
B_BOOT = 300          # simple decision-boundary bootstrap replicates
# Nested arms use 40 replicates rather than the 300 of the district arm. Each
# replicate retrains twelve ensembles and relabels a thousand districts, so the
# arms are not equally cheap. Forty is enough for what these arms are asked to
# establish, which is the ORDERING of the two variance components rather than a
# precise interval: the replicate contribution is smaller than the district
# contribution by orders of magnitude, not by a margin that forty versus sixty
# replicates would decide.
B_NESTED = 40         # nested bootstrap replicates (resample->retrain->relabel->refit)
B_JOINT = 40          # joint arm: districts AND replicates resampled together
S_MC = 400            # Monte-Carlo objective-propagation samples per population
S_MC_NESTED = 200     # lighter MC inside nested bootstrap

# QUICK smoke-test mode (set env MRMOU_QUICK=1). Reduces all heavy sizes so the
# full pipeline can be exercised end-to-end in seconds to catch runtime errors.
# This block MUST come after every size it overrides; when it was placed above
# the bootstrap sizes the B_BOOT and B_NESTED overrides were silently discarded
# and the smoke test ran the full 300 and 60 replicates.
import os as _os
if _os.environ.get("MRMOU_QUICK", "0") == "1":
    N_SAMPLE_VECTORS = 1500
    CLUSTER_GRID = [40, 80, 120]
    N_CENTROIDS = 80
    N_DRAWS = 12
    HORIZON_WEEKS = 60
    M_POPULATIONS = 120
    N_ENSEMBLE = 6
    B_BOOT = 30
    B_NESTED = 6
    B_JOINT = 6
    S_MC = 60
    S_MC_NESTED = 40

# ----------------------------------------------------------------------
# Coverage parameter ranges (proportions, 0 to 1)
# ----------------------------------------------------------------------
MR1_RANGE = (0.40, 0.98)
MR2_RANGE = (0.30, 0.95)
SIA_RANGE = (0.00, 0.95)
MR1_NATIONAL = 0.86
MR2_NATIONAL = 0.807

# ----------------------------------------------------------------------
# Transmission and natural history parameters
# ----------------------------------------------------------------------
R0_MEASLES_MEAN = 15.9
R0_MEASLES_SD = 1.6
R0_RUBELLA_MEAN = 6.0
R0_RUBELLA_SD = 0.8
LATENT_WEEKS = 1.43
INFECTIOUS_WEEKS = 1.14
VE1_MEAN = 0.85
VE1_SD = 0.03
VE2_MEAN = 0.97
VE2_SD = 0.01
VE_SIA_MEAN = 0.84
VE_SIA_SD = 0.03
BIRTH_RATE_WEEKLY = 0.018 / 52.0
POP_UNIT = 100000
SEED_INFECTIVES = 20

CFR_MEASLES_U5 = 0.012
CFR_MEASLES_5PLUS = 0.002
CRS_PER_RUBELLA_WCBA = 0.65
FRAC_RUBELLA_IN_WCBA_FIRST_TRI = 0.018

LIFE_EXPECTANCY = 73.0
MEAN_AGE_MEASLES_DEATH = 3.0
DW_MEASLES = 0.051
DUR_MEASLES = 10.0 / 365.0
DW_CRS = 0.40
DUR_CRS = 40.0
DISCOUNT = 0.03

# ----------------------------------------------------------------------
# Intervention definitions and unit costs (USD per targeted child)
# ----------------------------------------------------------------------
INTERVENTIONS = ["sms_reminder", "chw_outreach", "sia_campaign", "mr_six_month"]
INTERVENTION_LABELS = {
    "sms_reminder": "SMS reminder",
    "chw_outreach": "CHW outreach",
    "sia_campaign": "SIA campaign",
    "mr_six_month": "MR at six months",
}
C_FIXED = {
    "sms_reminder": 0.30,
    "chw_outreach": 2.20,
    "sia_campaign": 0.80,
    # Lowering the first dose to six months is a schedule change delivered
    # through existing routine contacts, so its fixed cost is the programmatic
    # cost of the extra visit and the additional dose, not a campaign. It is
    # cheap to field and, unlike outreach, it does not close the coverage gap at
    # all: it moves protection earlier for the children already being reached.
    "mr_six_month": 0.45,
}
SIA_REACH = 0.50
COST_MR_DOSE = 1.20
COST_DELIVERY_ROUTINE = 0.90
COST_DELIVERY_SIA = 1.35
COST_TREAT_MEASLES = 42.0
COST_TREAT_CRS = 3100.0

# ----------------------------------------------------------------------
# Intervention EFFECTIVENESS as LITERATURE-DERIVED STOCHASTIC ranges.
# Each intervention closes a share of the MR1 coverage gap; the share is a
# random variable with a mean and standard deviation from published effect
# sizes rather than a single fixed formula.
#   SMS reminders  : reminder/recall meta-analyses ~ 20-40% of the gap.
#   CHW outreach   : home visits / defaulter tracing ~ 45-75% of the gap,
#                    strongest at low baseline coverage.
#   SIA campaign   : one round reaches ~ 40-60% of remaining susceptibles.
# ----------------------------------------------------------------------
EFF_SMS_MEAN = 0.30
EFF_SMS_SD = 0.05
EFF_CHW_MEAN = 0.62
EFF_CHW_LOWCOV = 0.15
EFF_CHW_SD = 0.06
EFF_SIA_MEAN = 0.50
EFF_SIA_SD = 0.07

# ----------------------------------------------------------------------
# Decision-rule economics and sensitivity sweeps
# ----------------------------------------------------------------------
WTP_THRESHOLD = 500.0
WTP_GRID = [250.0, 500.0, 1000.0, 2600.0]
ICER_DALY_FLOOR = 5.0
DMIN_GRID = [1.0, 5.0, 20.0, 50.0]   # swept: the floor shapes the CVaR tail directly
BUDGET_CAP_PER_100K = 250000.0
BUDGET_GRID = [150000.0, 250000.0, 400000.0, float("inf")]
DELTA_GRID = [0.10, 0.20, 0.30]
CVAR_ALPHA_GRID = [0.80, 0.90, 0.95]
TARGET_GRID = {
    "herd_92": (0.92, 0.00),
    "transmission_scaled": (0.88, 0.06),
    "elimination_95": (0.95, 0.00),
}
DEFAULT_TARGET = "transmission_scaled"

# ----------------------------------------------------------------------
# Bangladesh administrative divisions (8) with REAL data.
#   pop2022       : BBS Census 2022 (PEC adjusted) division population.
#   urban_frac    : BBS Census 2022 division urban share.
#   under5_frac   : division under-5 share.
#   incid_per_mil : suspected measles incidence per million, 2026 outbreak
#                   (WHO SEARO situation report, 15 Mar - 14 Apr 2026).
# r0_mult and cfr_mult are DERIVED from real urbanicity and the 2026 incidence
# signal (see _division_derived).
# ----------------------------------------------------------------------
DIVISIONS_REAL = {
    #              pop2022      urban_frac  under5_frac  incid_per_mil
    "Dhaka":       (45643915,   0.435,      0.088,       35.0),
    "Chattogram":  (34178581,   0.315,      0.098,       19.8),
    "Khulna":      (17813957,   0.300,      0.078,       21.2),
    "Rajshahi":    (20794023,   0.271,      0.083,       28.3),
    "Rangpur":     (18020073,   0.230,      0.086,       17.5),
    "Barishal":    (9325818,    0.242,      0.084,       39.4),
    "Sylhet":      (11415021,   0.213,      0.101,       12.6),
    "Mymensingh":  (12637524,   0.196,      0.090,       29.6),
}
NATIONAL_POP_2022 = 169828911
NATIONAL_URBAN_FRAC = 0.3166


# ----------------------------------------------------------------------
# Derivation of the division transmission multiplier.
#
# The transmission multiplier blends two REAL signals on the log scale:
#   (i)  division urbanicity (Census 2022), a structural contact-rate proxy;
#   (ii) division measles incidence per million in the 2026 outbreak
#        (WHO SEARO), a realized-transmission signal.
# Observed outbreak incidence reflects BOTH transmissibility and accumulated
# susceptibility, so it is not a clean R0 measurement; the blend weight
# R0_MULT_W_INCID controls how much of the multiplier the incidence signal
# carries. R0_MULT_MODE selects the derivation used, and all three modes are
# swept in the sensitivity analysis so the dependence on this modelling choice
# is reported rather than assumed away.
# ----------------------------------------------------------------------
R0_MULT_MODE = "calibrated"     # one of: "calibrated", "blend", "urban_only", "incidence_only"
R0_MULT_W_INCID = 0.50          # weight on the incidence signal in "blend"
R0_MULT_SPREAD = 0.55           # total spread of the multiplier across divisions
R0_MULT_MODE_GRID = ["calibrated", "blend", "urban_only", "incidence_only"]
# Division scalings FITTED to the observed 2026 incidence by stage3_calibrate and
# persisted so every stage, in any process, uses the calibrated values rather
# than silently falling back to an assumption.
CALIBRATED_R0_MULT = {}


def _znorm(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-9)


def _division_derived(mode=None, w_incid=None):
    """Derive (under5_frac, urban_frac, r0_mult, cfr_mult) per division.

    r0_mult is a normalized blend of standardized urbanicity and standardized
    log incidence; cfr_mult scales inversely with urbanicity, reflecting poorer
    measles case management and nutritional status in less urban divisions.
    Both are centred so that the population weighted mean is one, so the blend
    changes the cross division CONTRAST without changing the national level.
    """
    mode = mode or R0_MULT_MODE
    w_incid = R0_MULT_W_INCID if w_incid is None else w_incid
    names = list(DIVISIONS_REAL.keys())
    if mode == "calibrated":
        # Scalings fitted so the model reproduces each division's observed 2026
        # incidence. If calibration has not run yet, fall back to the blend so
        # the module still imports; stage3 overwrites this.
        if CALIBRATED_R0_MULT:
            r0c = np.array([CALIBRATED_R0_MULT[d] for d in names], float)
            popc = np.array([DIVISIONS_REAL[d][0] for d in names], float)
            r0c = r0c / np.average(r0c, weights=popc)
            cfrc = 1.0 - R0_MULT_SPREAD * _znorm([DIVISIONS_REAL[d][1] for d in names]) / (
                2.0 * max(np.abs(_znorm([DIVISIONS_REAL[d][1] for d in names])).max(), 1e-9))
            cfrc = cfrc / np.average(cfrc, weights=popc)
            return {names[i]: (DIVISIONS_REAL[names[i]][2], DIVISIONS_REAL[names[i]][1],
                               float(round(r0c[i], 3)), float(round(cfrc[i], 3)))
                    for i in range(len(names))}
        mode = "blend"
    urb = np.array([DIVISIONS_REAL[d][1] for d in names])
    inc = np.array([DIVISIONS_REAL[d][3] for d in names])
    pop = np.array([DIVISIONS_REAL[d][0] for d in names], float)

    z_urb = _znorm(urb)
    z_inc = _znorm(np.log(inc))          # incidence is right skewed across divisions
    if mode == "urban_only":
        z = z_urb
    elif mode == "incidence_only":
        z = z_inc
    elif mode == "blend":
        z = (1.0 - w_incid) * z_urb + w_incid * z_inc
        z = _znorm(z)
    else:
        raise ValueError(mode)

    # map the standardized signal onto a multiplier with the intended spread
    r0 = 1.0 + R0_MULT_SPREAD * z / (2.0 * max(np.abs(z).max(), 1e-9)) * 2.0
    r0 = r0 / np.average(r0, weights=pop)
    cfr = 1.0 - R0_MULT_SPREAD * _znorm(urb) / (2.0 * max(np.abs(_znorm(urb)).max(), 1e-9))
    cfr = cfr / np.average(cfr, weights=pop)
    return {names[i]: (
        DIVISIONS_REAL[names[i]][2],
        DIVISIONS_REAL[names[i]][1],
        float(round(r0[i], 3)),
        float(round(cfr[i], 3)),
    ) for i in range(len(names))}


DIVISIONS = _division_derived()

_names = list(DIVISIONS_REAL.keys())
_u5_pop = np.array([DIVISIONS_REAL[d][0] * DIVISIONS_REAL[d][2] for d in _names])
DIVISION_U5_SHARE = {_names[i]: float(_u5_pop[i] / _u5_pop.sum()) for i in range(len(_names))}

AGE_COHORTS = ["6-11m", "12-23m", "24-35m", "36-47m", "48-59m"]
AGE_WEIGHTS_CENSUS = np.array([0.18, 0.24, 0.22, 0.19, 0.17])
SEX = ["male", "female"]
SEX_WEIGHTS_CENSUS = np.array([0.51, 0.49])

DATA_SOURCES = {
    "WUENIC": "WHO and UNICEF Estimates of National Immunization Coverage (annual).",
    "CES_2023": "Bangladesh Coverage Evaluation Survey 2023 (MR1 86%, MR2 80.7%).",
    "BBS_Census_2022": "Bangladesh Bureau of Statistics, Population and Housing Census 2022 (PEC adjusted division populations).",
    "WHO_DON598": "WHO Disease Outbreak News, Measles - Bangladesh, DON598, 4 April 2026.",
    "WHO_SEARO_2026": "WHO SEARO situation reports, Bangladesh measles outbreak, March-April 2026 (division incidence per million).",
    "Gavi": "Gavi, the Vaccine Alliance, MR vaccine price and delivery benchmarks.",
    "GBD_2019": "Global Burden of Disease 2019, measles and CRS disability weights.",
    "Guerra_2017": "Guerra et al. 2017, The basic reproduction number of measles: a systematic review.",
}


# ----------------------------------------------------------------------
# Age structured model inputs (epi_model_age.py)
# ----------------------------------------------------------------------
import json as _json
import os as _osmod

# Real Bangladesh age mixing matrices, collapsed from the published Prem et al.
# synthetic contact matrices onto this model's age groups. Built by
# tools/build_contact_matrix.py from the BGD national, urban and rural matrices.
with open(_osmod.path.join(_osmod.path.dirname(_osmod.path.dirname(_osmod.path.abspath(__file__))),
                       "data", "contact_matrix_bgd.json")) as _f:
    CONTACT_MATRICES = _json.load(_f)

# Maternal antibody waning. The literature anchors the shape: Guerra et al. 2018
# and the Lao PDR seroprevalence cohort (74 percent seropositive at 2 months, 28
# percent at 4 months, under 14 percent just before the 9 month dose). The RATE
# is calibrated here rather than taken from Lao, for a substantive reason: those
# cohorts were born to mothers with largely vaccine derived immunity, which
# transfers lower antibody titres and wanes faster, whereas Bangladeshi mothers
# of this cohort carry a larger share of natural immunity. The calibrated rate
# sits below the Lao estimate accordingly, and is reported as calibrated.
MATERNAL_WANE_MONTHLY = 0.24    # calibrated; see stage3_calibrate
MATERNAL_WANE_WEEKLY = MATERNAL_WANE_MONTHLY * 12.0 / 52.0
MATERNAL_IMMUNE_MOTHERS = 0.95      # share of births protected at birth
MATERNAL_PROT_0_5MO = 0.53          # implied by the waning rate over 0-5 months
MATERNAL_PROT_6_8MO = 0.17          # implied by the waning rate over 6-8 months

# First dose efficacy below 9 months. Source: Nic Lochlainn et al. 2019 (Lancet
# Infectious Diseases) systematic review and meta-analysis, which reports
# seroconversion of about 51 percent for a first dose given before 9 months
# against about 83 percent at or after 9 months. This single parameter carries
# the entire trade-off in the age lowering decision: a dose given at 6 months
# arrives before the infant window but takes far less often.
VE_EARLY_MEAN = 0.51
VE_EARLY_SD = 0.06

# Coverage history used to build the 2026 immunity profile. Sources: UNICEF
# reports MR1 declining from 88.6 percent in 2019 to 86 percent, and MR2 from 89
# percent in 2019 to 80.7 percent; the 2023 Coverage Evaluation Survey gives
# valid MR1 86.1 percent and MR2 80.7 percent. Note that WUENIC carries 97
# percent MR1 and 93 percent MR2 for the same period, a discrepancy the survey
# figures are preferred over and which Section 4 discusses.
HIST_MR1_2019 = 0.886
HIST_MR2_2019 = 0.890
HIST_SIA_REACH = 0.85               # 2014 and 2020 national campaigns
ADULT_IMMUNE_FRAC = 0.97            # vaccine-derived component before natural catch-up

# Natural infection catch-up. Measles has circulated continuously in Bangladesh,
# so older cohorts who escaped vaccination have largely been infected. These two
# parameters are the share of the residual vaccine susceptibles in each older
# group that natural infection has already removed. They are CALIBRATED (see
# stage3_calibrate) to the observed share of 2026 cases in children under five,
# because no direct serosurvey is available; the under nine month and under two
# year shares are then held out and used for validation.
NAT_INFECTION_HAZARD = 0.08      # per year; calibrated in stage3_calibrate
WUENIC_MR1_2023 = 0.97
WUENIC_MR2_2023 = 0.93

# Observed 2026 outbreak age distribution, used for VALIDATION and never fitted.
# Sources: WHO Disease Outbreak News DON598; UNICEF Situation Report No. 1; and the 2026
# situational analysis of Kamrujiaman et al., which reports 81 percent of cases under five
# including 34 percent in infants under nine months.
OBS_AGE_SHARE_UNDER9MO = 0.34
OBS_AGE_SHARE_UNDER2Y = 0.66
OBS_AGE_SHARE_UNDER5Y = 0.81

# Observation window of the reported division incidence (15 March to 14 April
# 2026), used for calibration.
CALIB_WINDOW_WEEKS = 4


# Load the fitted division scalings, if calibration has been run. This sits at the
# end of the module because it depends on the imports above, and it is what makes
# "calibrated" the real default rather than a label on a fallback.
_CALIB_PATH = _osmod.path.join(_osmod.path.dirname(_osmod.path.dirname(
    _osmod.path.abspath(__file__))), "data", "calibrated_r0_mult.json")
if _osmod.path.exists(_CALIB_PATH):
    with open(_CALIB_PATH) as _f:
        CALIBRATED_R0_MULT = _json.load(_f)
    DIVISIONS = _division_derived("calibrated")


# ----------------------------------------------------------------------
# Within district clustering of unvaccinated children
# ----------------------------------------------------------------------
# The single stratum model spreads the unvaccinated evenly across a district and
# therefore misses the held out share of cases under two years. They are not
# evenly spread: WHO and the 2026 situational analysis report 72 percent of cases
# as zero dose and 16 percent as partially vaccinated, against national first
# dose coverage of 86 percent, with cases concentrated in dense informal
# settlements in Dhaka. Under homogeneous coverage a first dose coverage of 0.86
# and an efficacy of 0.85 imply that only about 52 percent of cases would be zero
# dose; the gap between 52 and 72 percent is the clustering, and it is
# measurable, so we fit it rather than assume it.
#
# Each district is split into a reached stratum and an under-reached stratum.
# CLUSTER_FRAC is the population share of the under-reached stratum.
# CLUSTER_KAPPA is the strength of clustering: 0 gives both strata the district's
#   coverage (no clustering), 1 gives the under-reached stratum zero coverage.
# CLUSTER_ASSORT is the share of contacts made within one's own stratum in excess
#   of proportionate mixing; it is what makes clustering raise the attack rate
#   among the under-reached rather than merely relabel them.
CLUSTER_FRAC = 0.18
CLUSTER_KAPPA = 0.88      # calibrated to the 72 percent zero dose share of cases
CLUSTER_ASSORT = 0.90     # calibrated jointly with KAPPA
CLUSTER_ENABLED = False   # see stage3b_clustering_diagnostic

# Observed vaccination status of cases, used as a calibration target.
# Source: Kamrujiaman et al. 2026 situational analysis; WHO SEARO.
OBS_ZERO_DOSE_SHARE_OF_CASES = 0.72
OBS_PARTIAL_SHARE_OF_CASES = 0.16


# Cohort specific coverage decline.
# The model previously gave every age group the district's current coverage,
# which cannot be right and which is why it under-weighted the 9 to 23 month
# band. Coverage collapsed over time, so each cohort carries the coverage of the
# year it was vaccinated, not of today: children aged 9 to 23 months in March
# 2026 were due their doses in 2025 and early 2026, when the documented vaccine
# stockouts bit hardest and a catch-up campaign scheduled for 2024 had never been
# held, whereas children aged 24 to 59 months were vaccinated before the steepest
# part of the decline. Older cohorts are therefore BETTER covered than younger
# ones, which reverses the susceptibility ordering between those two bands.
# Coverage for a cohort of mean age a years is scaled by
#   g(a) = 1 - COHORT_COLLAPSE * exp(-a / COHORT_COLLAPSE_TAU)
# so the youngest vaccinated cohorts, whose doses were due during the stockout,
# carry the deepest shortfall and older cohorts approach the district's nominal
# coverage. The timescale is set by how long the disruption lasted: the catch-up
# campaign due in 2024 was never held and vaccine stocks were exhausted by early
# 2026, so the affected cohorts are roughly those under two years old in March
# 2026. Both parameters are calibrated.
COHORT_COLLAPSE = 0.0     # 0 disables; swept in the diagnostic
COHORT_COLLAPSE_TAU = 1.5   # years
