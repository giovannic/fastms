args = commandArgs(trailingOnly=TRUE)
node <- as.numeric(args[1])
global_fits <- args[2]
batch_size <- as.numeric(args[3])
n_batches <- as.numeric(args[4])
seed <- as.numeric(args[5])
warmup <- as.numeric(args[6])
daily_dir <- args[7]
yearly_dir <- args[8]

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

seasonality <- seasonality[sample(nrow(seasonality), n, replace = TRUE), ]

# nets
llins <- lapply(fits$interventions, function(i) head(i$llin, history_years))
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
irs <- lapply(fits$interventions, function(i) head(i$irs, history_years))
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
smc <- lapply(fits$interventions, function(i) head(i$smc, history_years))
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
    c(rep(0, history_years), active * runif(future_years, 0., 0.85))
  }
)

# tx
tx <- lapply(fits$interventions, function(i) head(i$tx, history_years))
tx <- tx[!duplicated(tx)]
tx <- tx[sample(seq(tx), n, replace = TRUE)]
tx <- lapply(
  tx,
  function(coverages) {
    c(coverages, runif(future_years, 0, 1))
  }
)

prop_act <- lapply(fits$interventions, function(i) head(i$prop_act, history_years))
prop_act <- prop_act[!duplicated(prop_act)]
prop_act <- prop_act[sample(seq(prop_act), n, replace = TRUE)]
prop_act <- lapply(
  prop_act,
  function(coverages) {
    c(coverages, rep(1, future_years))
  }
)

period <- history_years + future_years
one_round_timesteps <- seq(0, period - 1) * year
three_round_timesteps <- floor(seq(0, period - 1/3, by=1/3) * year)

process_row <- function(i) {
  row <- params[i,]
  seas_row <- seasonality[i,]
  parameters <- malariasimulation::get_parameters(list(
    human_population = 10000,
    individual_mosquitoes = FALSE,
    model_seasonality = TRUE,
    g0 = seas_row$seasonal_a0,
    g = c(seas_row$seasonal_a1, seas_row$seasonal_a2, seas_row$seasonal_a3),
    h = c(seas_row$seasonal_b1, seas_row$seasonal_b2, seas_row$seasonal_b3),
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
  
  # bednets
  parameters <- malariasimulation::set_bednets(
    parameters,
    timesteps = one_round_timesteps + (warmup * year),
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
    timesteps = one_round_timesteps + (warmup * year),
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
    timesteps = one_round_timesteps + (warmup * year),
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
    timesteps = three_round_timesteps + (warmup * year),
    coverages = rep(smc[[i]], each=3),
    min_age = .5 * year,
    max_age = 5 * year
  )

  # tx
  parameters <- malariasimulation::set_clinical_treatment(
    parameters,
    drug = 1,
    timesteps = one_round_timesteps + warmup * year,
    coverages = tx[[i]] * (1 - prop_act[[i]])
  )
  
  parameters <- malariasimulation::set_clinical_treatment(
    parameters,
    drug = 2,
    timesteps = one_round_timesteps + warmup * year,
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

to_dense_seq <- function(timesteps, values, range) {
  s <- rep(0, range)
  s[timesteps] <- values
  s
}

fillna <- function(v, value) {
  v[is.na(v)] <- value
  v
}

daily_parameters <- function(i) {
  species_vector <- unlist(lapply(
    mosquito_params[[i]],
    function(p) {
      p$species <- NULL
      as.numeric(p)
    }
  ))
  p <- params[i,]
  p$init_EIR <- NULL
  p$EIR <- mean(results[[i]]$EIR[seq((warmup - 1)*year, (warmup)*year)])
  c(
    as.numeric(p),
    species_vector,
    mosquito_proportions[i,]
  )
}

yearly_parameters <- function(i, rainfall) {
  c(daily_parameters(i), rainfall)
}

daily_timed <- function(i, rainfall) {
  matrix(
    c(
      rep(rainfall, period),
      to_dense_seq(one_round_timesteps + 1, llins[[i]], period * year),
      to_dense_seq(one_round_timesteps + 1, irs[[i]], period * year),
      to_dense_seq(three_round_timesteps + 1, rep(smc[[i]], each=3), period * year),
      to_dense_seq(one_round_timesteps + 1, rtss[[i]], period * year),
      rep(tx[[i]], each=year),
      rep(prop_act[[i]], each=year)
    ),
    ncol = 7,
    nrow = (history_years + future_years) * year
  )
}

yearly_timed <- function(i) {
  matrix(
    c(llins[[i]], irs[[i]], smc[[i]], rtss[[i]], tx[[i]], prop_act[[i]]),
    ncol = 6,
    nrow = history_years + future_years
  )
}

daily_outputs <- function(result) {
  matrix(
    c(
      result$n_detect_0_36500 / result$n_0_36500,
      fillna(result$n_inc_clinical_0_36500, 0) / result$n_0_36500,
      result$EIR
    ),
    ncol = 3,
    nrow = (history_years + future_years) * year
  )
}

yearly_outputs <- function(result) {
  o <- daily_outputs(result)
  apply(o, 2, function(series) colMeans(matrix(series, nrow=year)))
}

get_rainfall <- function(i) {
  seas_row <- seasonality[i,]
  vapply(
    1:365,
    function(t) malariasimulation:::rainfall(
      t,
      g0 = seas_row$seasonal_a0,
      g = c(seas_row$seasonal_a1, seas_row$seasonal_a2, seas_row$seasonal_a3),
      h = c(seas_row$seasonal_b1, seas_row$seasonal_b2, seas_row$seasonal_b3)
    ),
    numeric(1)
  )
}

format_daily <- function(i) {
  rainfall <- get_rainfall(i)
  result <- results[[i]][seq(warmup*year + 1, nrow(results[[i]])),]
  list(
    parameters = daily_parameters(i),
    timed_parameters = daily_timed(i, rainfall),
    outputs = daily_outputs(result)
  )
}

format_yearly <- function(i) {
  rainfall <- get_rainfall(i)
  result <- results[[i]][seq(warmup*year + 1, nrow(results[[i]])),]
  list(
    parameters = yearly_parameters(i, rainfall),
    timed_parameters = yearly_timed(i),
    outputs = yearly_outputs(result)
  )
}

for (batch_i in seq_along(batches)) {
  dailypath <- file.path(
    daily_dir,
    paste0('realisation_', node, '_batch_', batch_i, '.json')
  )
  yearlypath <- file.path(
    yearly_dir,
    paste0('realisation_', node, '_batch_', batch_i, '.json')
  )
  if (!file.exists(dailypath)) {
    start_time <- Sys.time()
    print(paste0('node ', node, ' batch ', batch_i, ' starting'))
    # do the work
    results <- lapply(batches[[batch_i]], process_row)

    daily_data <- lapply(batches[[batch_i]], format_daily)
    yearly_data <- lapply(batches[[batch_i]], format_yearly)

    jsonlite::write_json(daily_data, dailypath, auto_unbox=TRUE, pretty=TRUE)
    jsonlite::write_json(yearly_data, yearlypath, auto_unbox=TRUE, pretty=TRUE)
    print(paste0('node ', node, ' batch ', batch_i, ' completed'))
    print(Sys.time())
    print(Sys.time() - start_time)
  }
}
