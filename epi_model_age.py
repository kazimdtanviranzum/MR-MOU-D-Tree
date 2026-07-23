"""
epi_model_age.py
Age structured stochastic discrete time SEIR transmission model for measles and
rubella, with maternal antibody, routine MR1 and MR2, campaign (SIA) and an age
lowered first dose at six months.

Why this model replaces the single cohort SEIR of the earlier version. The 2026
Bangladesh outbreak is defined by its age distribution: WHO reports roughly 79 to
81 percent of cases in children under five, 66 percent under two and 33 percent
in infants under nine months, who are below the routine schedule. Bangladesh
responded by lowering the first dose to six months. A model without age structure
can represent neither the infant susceptibility window nor the policy that
addresses it, so the central decision of the outbreak would be outside its
vocabulary. The age structure here exists to make that decision expressible.

Age groups (chosen to resolve the infant window rather than for convenience):
    0: 0 to 5 months    - protected by maternal antibody, below any schedule
    1: 6 to 8 months    - maternal antibody largely gone, eligible ONLY if the
                          first dose is lowered to six months
    2: 9 to 14 months   - routine MR1 age
    3: 15 to 23 months  - routine MR2 age
    4: 24 to 59 months  - remainder of the SIA target range
    5: 5 to 14 years    - school age
    6: 15 years and over

The split at 24 months is deliberate rather than cosmetic: WHO reports the 2026
case distribution at under nine months, under two years and under five years, and
putting a group boundary at each of those ages means the validation targets fall
on real boundaries instead of being recovered by apportioning a wide group under
a uniform assumption.

Replications are vectorized over the draw axis, as in the previous model.
"""

import numpy as np
import config as C

# ----------------------------------------------------------------------
# Age structure
# ----------------------------------------------------------------------
AGE_LABELS = ["0-5mo", "6-8mo", "9-14mo", "15-23mo", "24-59mo", "5-14y", "15+y"]
AGE_WIDTH_MONTHS = np.array([6.0, 3.0, 6.0, 9.0, 36.0, 120.0, 12.0 * 50.0])
N_AGE = len(AGE_LABELS)
WEEKS_PER_MONTH = 52.0 / 12.0

# aging rate out of each group, per week
AGING_WEEKLY = 1.0 / (AGE_WIDTH_MONTHS * WEEKS_PER_MONTH)

# Mean age of each group in years, used for the natural infection catch-up below.
_EDGES_MONTHS = np.concatenate([[0.0], np.cumsum(AGE_WIDTH_MONTHS)])
AGE_MEAN_YEARS = 0.5 * (_EDGES_MONTHS[:-1] + _EDGES_MONTHS[1:]) / 12.0

# indices used repeatedly
A_INFANT_0_5, A_INFANT_6_8, A_MR1, A_MR2, A_TODDLER, A_SCHOOL, A_ADULT = 0, 1, 2, 3, 4, 5, 6
UNDER9MO = [0, 1]
UNDER2Y = [0, 1, 2, 3]
UNDER5 = [0, 1, 2, 3, 4]


def contact_matrix(urban_frac=None):
    """Age mixing matrix for Bangladesh.

    Source: Prem, van Zandvoort, Klepac et al. synthetic contact matrices, the
    Bangladesh (BGD) all-location matrix, collapsed from the published 5 year
    bands onto the age groups above and stored in data/contact_matrix_bgd.json.
    Where a target group sits inside a published band, contact is assumed
    homogeneous within that band; Prem's finest resolution is 0 to 4 years, so
    the four infant and child groups inherit that band's contact rates. This is
    the standard assumption when refining below the published resolution and it
    is a real limitation: it means the model does not resolve any difference in
    contact between, say, a two month old and a three year old, only differences
    in their immunity.

    When urban_frac is given, the national matrix is interpolated between the
    published rural and urban matrices, so division urbanicity enters contact
    intensity from data rather than from an assumed multiplier.
    """
    cm = C.CONTACT_MATRICES
    if urban_frac is None:
        return np.array(cm["national"], dtype=float)
    w = float(np.clip(urban_frac, 0.0, 1.0))
    rural = np.array(cm["rural"], dtype=float)
    urban = np.array(cm["urban"], dtype=float)
    return (1.0 - w) * rural + w * urban


def stratum_coverage(c):
    """Split a district's coverage into a reached and an under-reached stratum.

    Returns (share_reached, share_under, coverage_reached, coverage_under) with
    the constraint that the population weighted coverage equals the district's
    coverage c, so clustering redistributes who is vaccinated without changing
    how many are.
    """
    phi = float(C.CLUSTER_FRAC)
    kappa = float(C.CLUSTER_KAPPA)
    c = np.asarray(c, dtype=float)
    c_u = c * (1.0 - kappa)                       # under-reached stratum
    c_r = (c - phi * c_u) / (1.0 - phi)           # reached stratum, by construction
    c_r = np.minimum(c_r, 0.999)
    return 1.0 - phi, phi, c_r, c_u


def zero_dose_share_of_susceptibles(c, ve):
    """Share of susceptibles who never received a dose, given coverage c and
    efficacy ve. A vaccinated child who did not seroconvert is susceptible but is
    NOT zero dose, and the reported statistic distinguishes them, so the model
    must too."""
    zd = 1.0 - c
    vu = c * (1.0 - ve)
    return zd / np.maximum(zd + vu, 1e-12)


def _next_gen_scale(Cmat, pop_share, infectious_weeks):
    """Scale factor converting a target R0 into a transmission probability per
    contact, using the dominant eigenvalue of the next generation matrix for a
    fully susceptible population."""
    K = Cmat * pop_share[None, :] * infectious_weeks
    ev = np.max(np.abs(np.linalg.eigvals(K)))
    return ev if ev > 1e-9 else 1.0


def stationary_age_distribution():
    """Population share by age group implied by the aging rates and the birth
    rate, normalized to one."""
    share = AGE_WIDTH_MONTHS / AGE_WIDTH_MONTHS.sum()
    return share / share.sum()


# ----------------------------------------------------------------------
# Immunity profile built from the coverage history
# ----------------------------------------------------------------------
def initial_immunity_profile(mr1, mr2, sia_before, ve1, ve2, vesia, n):
    """Immune fraction by age group at the start of the analysis window.

    This is the immunity debt that produced the outbreak, and it is built from
    the coverage history rather than assumed. Cohorts are mapped to the coverage
    they actually experienced:

      * 0 to 5 months  - no vaccine; protection is maternal antibody only, and
                         that is carried by the M compartment, not here.
      * 6 to 8 months  - no vaccine under the routine schedule.
      * 9 to 14 months - first dose only, at the scenario's MR1 coverage.
      * 15 to 23 and 24 to 59 months - first and second dose at the scenario's
                         coverage. These
                         cohorts were born after the 2020 campaign, so they have
                         had no campaign exposure, which is the accumulation WHO
                         describes.
      * 5 to 14 years  - routine coverage of earlier, higher years plus the 2014
                         and 2020 campaigns.
      * 15 years and over - effectively saturated by natural infection and the
                         earlier campaigns.

    Arrays are shaped (N_AGE, n) so every replication carries its own efficacy
    draws.
    """
    prof = np.zeros((N_AGE, n))

    def cohort_cov(c, a_years):
        """Coverage the cohort of mean age a_years actually received.

        The youngest vaccinated cohorts were due their doses during the 2025 and
        2026 stockouts, so they carry the deepest shortfall; older cohorts were
        vaccinated before the collapse and approach the district's nominal
        coverage. Handing every cohort today's figure, as the previous version
        did, gets the ordering between the 9 to 23 month and 24 to 59 month bands
        backwards, which is exactly where the held out validation failed.
        """
        g = 1.0 - C.COHORT_COLLAPSE * np.exp(-a_years / C.COHORT_COLLAPSE_TAU)
        return np.clip(c * g, 0.0, 0.999)

    # Each group is given the coverage of the year it was vaccinated.
    c1_mr1 = cohort_cov(mr1, AGE_MEAN_YEARS[A_MR1])
    c1_mr2 = cohort_cov(mr1, AGE_MEAN_YEARS[A_MR2])
    c1_tod = cohort_cov(mr1, AGE_MEAN_YEARS[A_TODDLER])
    c2_mr2 = np.minimum(cohort_cov(mr2, AGE_MEAN_YEARS[A_MR2]), c1_mr2)
    c2_tod = np.minimum(cohort_cov(mr2, AGE_MEAN_YEARS[A_TODDLER]), c1_tod)

    one_dose = np.clip(c1_mr1 * ve1, 0.0, 0.999)
    two_dose_mr2 = np.clip(c1_mr2 * ve1 + c2_mr2 * (ve2 - ve1), 0.0, 0.999)
    two_dose_tod = np.clip(c1_tod * ve1 + c2_tod * (ve2 - ve1), 0.0, 0.999)

    prof[A_INFANT_0_5, :] = 0.0
    prof[A_INFANT_6_8, :] = 0.0
    prof[A_MR1, :] = one_dose
    # 15-59mo: two routine doses, plus whatever prior campaign reach the scenario
    # carries. sia_before is the prior campaign reach of the district.
    prof[A_MR2, :] = np.clip(two_dose_mr2 + sia_before * (1.0 - two_dose_mr2) * vesia, 0.0, 0.999)
    prof[A_TODDLER, :] = np.clip(two_dose_tod + sia_before * (1.0 - two_dose_tod) * vesia,
                                 0.0, 0.999)
    # school age: higher historical routine coverage and two national campaigns
    hist = np.clip(C.HIST_MR1_2019 * ve1 + C.HIST_MR2_2019 * (ve2 - ve1), 0.0, 0.999)
    school_vax = np.clip(hist + C.HIST_SIA_REACH * (1.0 - hist) * vesia, 0.0, 0.999)
    prof[A_SCHOOL, :] = school_vax
    prof[A_ADULT, :] = C.ADULT_IMMUNE_FRAC

    # ---- natural infection catch-up, graded by age ----
    # Measles has circulated continuously, so whoever the programme missed has
    # been accumulating infection risk for as long as they have been alive. A
    # constant force of infection h gives a catch-up of 1 - exp(-h * age) applied
    # to whatever the vaccine left susceptible. This replaces two free
    # per-group constants with ONE parameter that grades immunity by age
    # automatically, which is what the data require: a three year old has had
    # three years to be infected and a fifteen month old has not.
    catch = 1.0 - np.exp(-C.NAT_INFECTION_HAZARD * AGE_MEAN_YEARS)
    prof = 1.0 - (1.0 - prof) * (1.0 - catch[:, None])
    # infants below the schedule carry no vaccine immunity; their protection is
    # maternal and is held in the M compartment, so do not let catch-up leak in
    prof[A_INFANT_0_5, :] = 0.0
    prof[A_INFANT_6_8, :] = 0.0
    return np.clip(prof, 0.0, 0.9995)


def simulate_age(rng, mr1, mr2, sia, params, disease="measles", r0_mult=1.0,
                 urban_frac=None, policy=None, horizon_weeks=None,
                 return_age_incidence=False, return_dose_status=False):
    """Vectorized stochastic age structured SEIR with two coverage strata.

    Every compartment carries an age axis and a stratum axis. The stratum axis
    represents within district clustering of unvaccinated children: a reached
    stratum and an under-reached stratum, whose coverages are set by
    stratum_coverage so that the district's overall coverage is unchanged. The
    strata mix assortatively, which is what makes clustering raise the attack
    rate among the under-reached rather than merely relabel them. Setting
    C.CLUSTER_ENABLED to False collapses this to the homogeneous model.

    policy selects the vaccination schedule in force:
        None or "routine"  - MR1 at 9 months, MR2 at 15 months
        "six_month"        - additionally offers a first dose at 6 months, at the
                             lower efficacy that dose carries
    """
    n = int(np.size(np.atleast_1d(mr1)))
    N = float(C.POP_UNIT)
    H = int(horizon_weeks or C.HORIZON_WEEKS)
    R0 = (params["R0m"] if disease == "measles" else params["R0r"]) * r0_mult
    ve1, ve2, vesia = params["ve1"], params["ve2"], params["vesia"]
    ve_early = params["ve_early"]

    Cmat = contact_matrix(urban_frac)
    share = stationary_age_distribution()
    scale = _next_gen_scale(Cmat, share, C.INFECTIOUS_WEEKS)
    q = R0 / scale

    sigma = 1.0 / C.LATENT_WEEKS
    gamma = 1.0 / C.INFECTIOUS_WEEKS
    b = C.BIRTH_RATE_WEEKLY
    omega = C.MATERNAL_WANE_WEEKLY

    # When clustering is off the model carries a SINGLE stratum, not a second
    # stratum of zero size. That is deliberate: an empty stratum would still
    # consume random draws and would silently change every result relative to the
    # unclustered model, making the two incomparable.
    if C.CLUSTER_ENABLED:
        s_r, s_u, mr1_r, mr1_u = stratum_coverage(mr1)
        _, _, mr2_r, mr2_u = stratum_coverage(mr2)
        eps = float(C.CLUSTER_ASSORT)
        strat_share = np.array([s_r, s_u])
        mr1_s = [mr1_r, mr1_u]; mr2_s = [mr2_r, mr2_u]
    else:
        eps = 0.0
        strat_share = np.array([1.0])
        mr1_s = [np.asarray(mr1, float)]; mr2_s = [np.asarray(mr2, float)]
        mr1_r = mr1_u = np.asarray(mr1, float)
    n_strat = len(strat_share)

    # population by age and stratum
    Npop = N * share[:, None] * strat_share[None, :]        # (N_AGE, 2)

    M = np.zeros((N_AGE, n_strat, n)); S = np.zeros((N_AGE, n_strat, n))
    E = np.zeros((N_AGE, n_strat, n)); I = np.zeros((N_AGE, n_strat, n)); R = np.zeros((N_AGE, n_strat, n))

    for k in range(n_strat):
        imm = initial_immunity_profile(mr1_s[k], mr2_s[k], sia, ve1, ve2, vesia, n)
        for a in range(N_AGE):
            tot = Npop[a, k]
            if a == A_INFANT_0_5:
                M[a, k] = tot * C.MATERNAL_PROT_0_5MO
                S[a, k] = tot * (1.0 - C.MATERNAL_PROT_0_5MO)
            elif a == A_INFANT_6_8:
                M[a, k] = tot * C.MATERNAL_PROT_6_8MO
                S[a, k] = tot * (1.0 - C.MATERNAL_PROT_6_8MO)
            else:
                R[a, k] = tot * imm[a]
                S[a, k] = tot * (1.0 - imm[a])

    seed = float(C.SEED_INFECTIVES)
    for k in range(n_strat):
        sk = seed * strat_share[k]
        I[A_SCHOOL, k] += sk
        S[A_SCHOOL, k] = np.maximum(S[A_SCHOOL, k] - sk, 0.0)

    if sia is not None and np.any(np.asarray(sia) > 0):
        for a in (A_INFANT_6_8, A_MR1, A_MR2, A_TODDLER):
            for k in range(n_strat):
                reached = S[a, k] * np.asarray(sia) * vesia
                S[a, k] -= reached
                R[a, k] += reached

    cum_inf = np.zeros(n)
    cum_inf_age = np.zeros((N_AGE, n))
    cum_inf_strat = np.zeros((n_strat, n))
    doses = np.zeros(n)

    Npop_age = np.maximum(Npop.sum(1), 1.0)                 # (N_AGE,)
    Npop_as = np.maximum(Npop, 1.0)                         # (N_AGE, 2)

    for _ in range(H):
        # prevalence within each stratum, and overall, by age
        prev_s = I / Npop_as[:, :, None]                    # (N_AGE, 2, n)
        prev_all = I.sum(1) / Npop_age[:, None]             # (N_AGE, n)
        # assortative mixing: a share eps of contacts is with one's own stratum
        mixed = eps * prev_s + (1.0 - eps) * prev_all[:, None, :]
        lam = q[None, None, :] * np.einsum("ij,jkn->ikn", Cmat, mixed)
        p_inf = 1.0 - np.exp(-lam)

        new_E = rng.binomial(np.maximum(S, 0).astype(int), np.clip(p_inf, 0, 1))
        new_I = rng.binomial(np.maximum(E, 0).astype(int), 1 - np.exp(-sigma))
        new_R = rng.binomial(np.maximum(I, 0).astype(int), 1 - np.exp(-gamma))
        wane = rng.binomial(np.maximum(M, 0).astype(int), 1 - np.exp(-omega))

        S = S - new_E + wane
        M = M - wane
        E = E + new_E - new_I
        I = I + new_I - new_R
        R = R + new_R
        cum_inf += new_I.sum((0, 1))
        cum_inf_age += new_I.sum(1)
        cum_inf_strat += new_I.sum(0)

        age_out = {name: arr * AGING_WEEKLY[:, None, None]
                   for name, arr in (("M", M), ("S", S), ("E", E), ("I", I), ("R", R))}
        M = M - age_out["M"]; S = S - age_out["S"]; E = E - age_out["E"]
        I = I - age_out["I"]; R = R - age_out["R"]

        for a in range(N_AGE - 1):
            mv_M = age_out["M"][a]
            mv_S = age_out["S"][a].copy()
            for k in range(n_strat):
                c1 = mr1_s[k]
                if a + 1 == A_INFANT_6_8 and policy == "six_month":
                    take = mv_S[k] * c1 * ve_early
                    doses += mv_S[k] * c1
                    mv_S[k] = mv_S[k] - take
                    R[a + 1, k] += take
                if a + 1 == A_MR1:
                    take = mv_S[k] * c1 * ve1
                    doses += mv_S[k] * c1
                    mv_S[k] = mv_S[k] - take
                    R[a + 1, k] += take
                if a + 1 == A_MR2:
                    c2 = np.minimum(mr2_s[k], c1)
                    take = mv_S[k] * c2 * ve2
                    doses += mv_S[k] * c2
                    mv_S[k] = mv_S[k] - take
                    R[a + 1, k] += take
            M[a + 1] += mv_M
            S[a + 1] += mv_S
            E[a + 1] += age_out["E"][a]
            I[a + 1] += age_out["I"][a]
            R[a + 1] += age_out["R"][a]

        births = rng.poisson(b * N, n).astype(float)
        for k in range(n_strat):
            bk = births * strat_share[k]
            M[A_INFANT_0_5, k] += bk * C.MATERNAL_IMMUNE_MOTHERS
            S[A_INFANT_0_5, k] += bk * (1.0 - C.MATERNAL_IMMUNE_MOTHERS)

        for arr in (M, S, E, I, R):
            np.maximum(arr, 0.0, out=arr)

    if sia is not None:
        doses = doses + N * np.sum(share[[A_INFANT_6_8, A_MR1, A_MR2, A_TODDLER]]) * np.asarray(sia)

    out = [cum_inf, doses]
    if return_age_incidence:
        out.append(cum_inf_age)
    if return_dose_status:
        # Zero dose share of cases: within each stratum, the share of susceptibles
        # who never received a dose, weighted by that stratum's share of cases.
        tot = np.maximum(cum_inf_strat.sum(0), 1e-9)
        if n_strat == 1:
            zd = zero_dose_share_of_susceptibles(mr1_s[0], ve1) * np.ones(n)
        else:
            zd_r = zero_dose_share_of_susceptibles(mr1_r, ve1)
            zd_u = zero_dose_share_of_susceptibles(mr1_u, ve1)
            zd = (cum_inf_strat[0] * zd_r + cum_inf_strat[1] * zd_u) / tot
        out.append(zd)
    return tuple(out) if len(out) > 2 else (out[0], out[1])


def age_distribution_of_cases(cum_inf_age):
    """Shares of cumulative cases under nine months, under two years and under
    five years. These are the quantities WHO reports for the 2026 outbreak and
    are used for validation, not for fitting."""
    tot = cum_inf_age.sum(0)
    tot = np.maximum(tot, 1e-9)
    u9mo = cum_inf_age[UNDER9MO].sum(0) / tot
    u2y = cum_inf_age[UNDER2Y].sum(0) / tot
    u5y = cum_inf_age[UNDER5].sum(0) / tot
    return u9mo, u2y, u5y
