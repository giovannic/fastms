args = commandArgs(trailingOnly=TRUE)
node <- as.numeric(args[1])
global_fits <- args[2]
batch_size <- as.numeric(args[3])
n_batches <- as.numeric(args[4])
seed <- as.numeric(args[5])
warmup <- as.numeric(args[6])
out_dir <- args[7]

print(paste0('beginning node ', node))

history_years <- 19
future_years <- 20
year <- 365
month <- 30

fits <- readRDS(global_fits)

set.seed(seed)

db <- 10
dc <- 30
a0 <- 2920
rho<- 0.85
immunity_scale <- function(a, d) {
  (1 + rho / (d / a0 - 1) * exp(-a /a0) - (1 + rho/(d / a0 + 1)) * exp(-a / d))
}

n <- n_batches * batch_size
seasonality <- do.call(rbind, fits$seasonality)
seasonality <- unique(seasonality)
params <- data.frame(
  average_age = runif(n, 20 * 365, 40 * 365),
  init_EIR = runif(n, 0, 1000),
  sigma_squared = runif(n, 1, 3),
  du = runif(n, 30, 100),
  kb = runif(n, 0.01, 10),
  ub = runif(n, 1, 1000),
  uc = runif(n, 1, 1000), # not documented
  ud = runif(n, 1, 1000), # not documented
  kc = runif(n, 0.01, 10),
  b0 = runif(n, 0.01, 0.99),
  ct = runif(n, 0, 1),
  cd = runif(n, 0, 1),
  #ca = runif(n, 0, 1), #TODO: how is this changed?
  cu = runif(n, 0, 1)
)

daily_EIR <- params$init_EIR / 365

params$ib0 <- daily_EIR * db / (daily_EIR * params$ub + 1) * vapply(
  runif(n, 1, 30) * 365,
  function(a) immunity_scale(a, db),
  numeric(1)
)
params$ic0 <- daily_EIR * dc * vapply(
  runif(n, 1, 30) * 365,
  function(a) immunity_scale(a, dc),
  numeric(1)
)

params$b1 <- runif(n, 0, 1) * params$b0

mosquito_params <- lapply(
  seq(n),
  function(.) {
    lapply(
      c('gamb', 'arab', 'fun'),
      function(label) {
        f <- 1 / runif(1, 1, 4)
        list(
          species = label,
          mum = 1 / runif(1, 3, 20),
          blood_meal_rates = f,
          foraging_time = runif(1, 0, f),
          Q0 = runif(1, 0, 1),
          phi_indoors = runif(1, 0, 1),
          phi_bednets = runif(1, 0, 1)
        )
      }
    )
  }
)

mosquito_proportions <- matrix(runif(n * 3), nrow = n, ncol = 3)
mosquito_proportions <- mosquito_proportions / rowSums(mosquito_proportions)

params <- cbind(
  params,
  seasonality[sample(nrow(seasonality), n, replace = TRUE), ]
)

# nets
llins <- lapply(fits$interventions, function(i) i$llin)
llins <- llins[!duplicated(llins)]
llins <- llins[sample(seq(llins), n, replace = TRUE)]
llins <- lapply(
  llins,
  function(coverages) {
    c(coverages, runif(future_years, 0, 0.8))
  }
)
llins <- lapply(
  llins,
  function(l) {
    tail(
      as.numeric(
        filter( # smooth out the coverages (3 years)
          c(0, 0, l),
          rep(1 / 3, 3),
          sides = 1
        )
      ),
      -2
    )
  }
)

# irs
# NOTE: assuming only one round per year!
irs <- lapply(fits$interventions, function(i) i$irs)
irs <- irs[!duplicated(irs)]
irs <- irs[sample(seq(irs), n, replace = TRUE)]
irs <- lapply(
  irs,
  function(coverages) {
    active <- runif(future_years, 0, 1) > .5
    c(coverages, active * runif(future_years, 0., 0.85))
  }
)

# smc
# NOTE: assuming 3 rounds per year!
smc <- lapply(fits$interventions, function(i) i$smc)
smc <- smc[!duplicated(smc)]
smc <- smc[sample(seq(smc), n, replace = TRUE)]
smc <- lapply(
  smc,
  function(coverages) {
    active <- runif(future_years, 0, 1) > .5
    c(coverages, active * runif(future_years, 0., 0.85))
  }
)

# rtss
rtss <- lapply(
  seq(n),
  function(.) {
    active <- runif(future_years, 0, 1) > .5
    active * runif(future_years, 0., 0.85)
  }
)

# tx
tx <- lapply(fits$interventions, function(i) i$tx)
tx <- tx[!duplicated(tx)]
tx <- tx[sample(seq(tx), n, replace = TRUE)]
tx <- lapply(
  tx,
  function(coverages) {
    c(coverages, runif(future_years, 0, 1))
  }
)

prop_act <- lapply(fits$interventions, function(i) i$prop_act)
prop_act <- prop_act[!duplicated(prop_act)]
prop_act <- prop_act[sample(seq(prop_act), n, replace = TRUE)]
prop_act <- lapply(
  prop_act,
  function(coverages) {
    c(coverages, rep(1, future_years))
  }
)

process_row <- function(i) {
  row <- params[i,]
  parameters <- malariasimulation::get_parameters(list(
    human_population = 10000,
    individual_mosquitoes = FALSE,
    model_seasonality = TRUE,
    g0 = row$seasonal_a0,
    g = c(row$seasonal_a1, row$seasonal_a2, row$seasonal_a3),
    h = c(row$seasonal_b1, row$seasonal_b2, row$seasonal_b3),
    average_age = row$average_age,
    sigma_squared = row$sigma_squared,
    du = row$du,
    kb = row$kb,
    ub = row$ub,
    uc = row$uc,
    ud = row$ud,
    kc = row$kc,
    b0 = row$b0,
    b1 = row$b1,
    ib0 = row$ib0,
    ic0 = row$ic0,
    ct = row$ct,
    cd = row$cd,
    #ca = row$ca, #TODO: as above
    cu = row$cu,
    prevalence_rendering_min_ages = 0,
    prevalence_rendering_max_ages = 100 * year,
    clinical_incidence_rendering_min_ages = 0,
    clinical_incidence_rendering_max_ages = 100 * year
  ))

  parameters <- malariasimulation::set_species(
    parameters,
    mosquito_params[[i]],
    proportions = mosquito_proportions[i,]
  )

  parameters <- malariasimulation::set_drugs(
    parameters,
    list(
      malariasimulation::DHA_PQP_params,
      malariasimulation::AL_params,
      malariasimulation::SP_AQ_params
    )
  )

  parameters <- malariasimulation::set_equilibrium(parameters, row$init_EIR)
  
  period <- history_years + future_years

  # bednets
  parameters <- malariasimulation::set_bednets(
    parameters,
    timesteps = seq(0, period - 1) * year + (warmup * year),
    coverages = llins[[i]],
    retention = 5 * year,
    dn0 = matrix(.533, nrow=period, ncol=3),
    rn = matrix(.56, nrow=period, ncol=3),
    rnm = matrix(.24, nrow=period, ncol=3),
    gamman = rep(2.64 * year, period)
  )

  # spraying
	parameters<- malariasimulation::set_spraying(
		parameters,
    timesteps = seq(0, period - 1) * year + (warmup * year),
		coverages = irs[[i]],
		ls_theta = matrix(2.025, nrow=period, ncol=3),
		ls_gamma = matrix(-0.009, nrow=period, ncol=3),
		ks_theta = matrix(-2.222, nrow=period, ncol=3),
		ks_gamma = matrix(0.008, nrow=period, ncol=3),
		ms_theta = matrix(-1.232, nrow=period, ncol=3),
		ms_gamma = matrix(-0.009, nrow=period, ncol=3)
	)

  # rtss
	parameters <- malariasimulation::set_mass_rtss(
		parameters,
		timesteps = seq(0, future_years - 1) * year + ((warmup + history_years) * year),
		coverages = rtss[[i]],
		min_wait = 0,
		min_ages = 0,
		max_ages = 100 * year,
		boosters = 18 * month,
		booster_coverage = .8
	)

  # smc
  delivery_offset <- 30
  peak_offset <- malariasimulation::peak_season_offset(parameters) 
  parameters <- malariasimulation::set_smc(
    parameters,
    drug = 3,
    timesteps = floor(seq(0, period - 1/3, by=1/3) * year + (warmup * year)),
    coverages = rep(smc[[i]], each=3),
    min_age = .5 * year,
    max_age = 5 * year
  )

  # tx
  parameters <- malariasimulation::set_clinical_treatment(
    parameters,
    drug = 1,
    timesteps = seq(0, period - 1) * year + warmup * year,
    coverages = tx[[i]] * (1 - prop_act[[i]])
  )
  
  parameters <- malariasimulation::set_clinical_treatment(
    parameters,
    drug = 2,
    timesteps = seq(0, period - 1) * year + warmup * year,
    coverages = tx[[i]] * prop_act[[i]]
  )

  print(paste('row', i))

  malariasimulation::run_simulation(
    (period + warmup) * year,
    parameters=parameters
  )[c('n_detect_0_36500', 'n_0_36500', 'n_inc_clinical_0_36500', 'EIR')]
}

batches <- split(
  seq(nrow(params)),
  (seq(nrow(params))-1) %/% batch_size
)

for (batch_i in seq_along(batches)) {
  outpath <- file.path(
    out_dir,
    paste0('realisation_', node, '_batch_', batch_i, '.json')
  )
  if (!file.exists(outpath)) {
    start_time <- Sys.time()
    print(paste0('node ', node, ' batch ', batch_i, ' starting'))
    # do the work
    results <- lapply(batches[[batch_i]], process_row)

    outdata <- params[batches[[batch_i]],]
    outdata$mosquito_params <- mosquito_params[batches[[batch_i]]]
    outdata$mosquito_proportions <- apply(mosquito_proportions[batches[[batch_i]],,drop=F], 1, as.list)
    outdata$llin <- llins[batches[[batch_i]]]
    outdata$irs <- irs[batches[[batch_i]]]
    outdata$smc <- smc[batches[[batch_i]]]
    outdata$rtss <- rtss[batches[[batch_i]]]
    outdata$tx <- tx[batches[[batch_i]]]
    outdata$prop_act <- prop_act[batches[[batch_i]]]
    outdata$inc <- lapply(results, function(x) x$n_inc_clinical_0_36500 / x$n_0_36500)
    outdata$prev <- lapply(results, function(x) x$n_detect_0_36500 / x$n_0_36500)
    outdata$eir <- lapply(results, function(x) x$EIR)

    jsonlite::write_json(outdata, outpath, auto_unbox=TRUE, pretty=TRUE)
    print(paste0('node ', node, ' batch ', batch_i, ' completed'))
    print(Sys.time())
    print(Sys.time() - start_time)
  }
}
