args = commandArgs(trailingOnly=TRUE)
node <- as.numeric(args[1])
global_fits <- args[2]
batch_size <- as.numeric(args[3])
n_batches <- as.numeric(args[4])
seed <- as.numeric(args[5])
warmup <- as.numeric(args[6])
yearly_dir <- args[7]

print(paste0('beginning node ', node))

history_years <- 19
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
llins <- lapply(seq(n), function(.) runif(history_years, 0, 0.8))
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

# rtss
rtss <- lapply(seq(n), function(.) runif(history_years, 0., 0.85))

# tx
tx <- lapply(seq(n), function(.) runif(history_years, 0., 0.85))

prop_act <- lapply(seq(n), function(.) runif(history_years, 0., 0.85))

period <- history_years
one_round_timesteps <- seq(0, period - 1) * year

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
    prevalence_rendering_min_ages = c(0, 2 * year, 6 * month),
    prevalence_rendering_max_ages = c(100 * year, 10 * year, 59 * month)
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
  )[c(
    'n_detect_0_36500',
    'n_0_36500',
    'n_detect_730_3650',
    'n_730_3650',
    'n_detect_180_1770',
    'n_180_1770',
    'EIR'
  )]
}

batches <- split(
  seq(nrow(params)),
  (seq(nrow(params))-1) %/% batch_size
)

daily_parameters <- function(i, result) {
  species_vector <- unlist(lapply(
    mosquito_params[[i]],
    function(p) {
      p$species <- NULL
      as.numeric(p)
    }
  ))
  row <- params[i,]
  c(
    mean(result$EIR[seq((warmup - 1)*year, (warmup)*year)]),
    row$average_age,
    row$sigma_squared,
    row$du,
    row$kb,
    row$ub,
    row$uc,
    row$ud,
    row$kc,
    row$b0,
    row$b1,
    row$ct,
    row$cd,
    row$cu,
    species_vector,
    mosquito_proportions[i,]
  )
}

yearly_parameters <- function(i, result, rainfall) {
  c(daily_parameters(i, result), rainfall)
}

yearly_timed <- function(i) {
  matrix(
    c(llins[[i]], rtss[[i]], tx[[i]], prop_act[[i]]),
    ncol = 4,
    nrow = period
  )
}

daily_outputs <- function(result) {
  result <- result[seq(warmup*year + 1, nrow(result)),]
  matrix(
    c(
      result$n_detect_0_36500 / result$n_0_36500,
      result$n_detect_730_3650 / result$n_730_3650,
      result$n_detect_180_1770 / result$n_detect_180_1770,
      result$EIR
    ),
    ncol = 4,
    nrow = period * year
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

format_yearly <- function(i, result) {
  rainfall <- get_rainfall(i)
  list(
    parameters = yearly_parameters(i, result, rainfall),
    timed_parameters = yearly_timed(i),
    outputs = yearly_outputs(result)
  )
}

for (batch_i in seq_along(batches)) {
  yearlypath <- file.path(
    yearly_dir,
    paste0('realisation_', node, '_batch_', batch_i, '.json')
  )
  if (!file.exists(yearlypath)) {
    start_time <- Sys.time()
    print(paste0('node ', node, ' batch ', batch_i, ' starting'))
    # do the work
    results <- lapply(batches[[batch_i]], process_row)

    yearly_data <- lapply(
      seq_along(results),
      function(i) format_yearly(batches[[batch_i]][[i]], results[[i]])
    )

    jsonlite::write_json(yearly_data, yearlypath, auto_unbox=TRUE, pretty=TRUE)
    print(paste0('node ', node, ' batch ', batch_i, ' completed'))
    print(Sys.time())
    print(Sys.time() - start_time)
  }
}
