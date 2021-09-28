args = commandArgs(trailingOnly=TRUE)
node <- as.numeric(args[1])
global_fits <- args[2]
batch_size <- as.numeric(args[3])
n_batches <- as.numeric(args[4])
seed <- as.numeric(args[5])
warmup <- as.numeric(args[6])
yearly_dir <- args[7]

print(paste0('beginning node ', node))

period <- 16
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


r <- lhs::randomLHS(n, 17)
n_params <- 13


seasonality <- do.call(rbind, fits$seasonality)
seasonality <- unique(seasonality)
params <- data.frame(
  average_age = qunif(r[,1], min=20 * 365, max=40 * 365),
  init_EIR = qunif(r[,2], min=0, max=1000),
  sigma_squared = qunif(r[,3], min=1, max=3),
  du = qunif(r[,3], min=30, max=100),
  kb = qunif(r[,4], min=0.01, max=10),
  ub = qunif(r[,5], min=1, max=1000),
  uc = qunif(r[,6], min=1, max=1000), # not documented
  ud = qunif(r[,7], min=1, max=1000), # not documented
  kc = qunif(r[,8], min=0.01, max=10),
  b0 = qunif(r[,9], min=0.01, max=0.99),
  ct = qunif(r[,10], min=0, max=1),
  cd = qunif(r[,11], min=0, max=1),
  #ca = runif(n, 0, 1), #TODO: how is this changed?
  cu = qunif(r[,12], min=0, max=1)
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

params$b1 <- qunif(r[,13], min=0, max=1) * params$b0

species <- c('gamb', 'arab', 'fun')
mosquito_params <- lapply(
  seq(n),
  function(i) {
    lapply(
      seq_along(species),
      function(j) {
        list(
          species = species[[j]],
          mum = 0.1253333,
          blood_meal_rates = 1 / 3,
          foraging_time = 0.69,
          Q0 = qunif(r[i,12 + j], min=0, max=1),
          phi_indoors = qunif(r[i, 12 + j], min=0, max=1),
          phi_bednets = qunif(r[i, 12 + j], min=0, max=1)
        )
      }
    )
  }
)

mosquito_proportions <- matrix(runif(n * 3), nrow = n, ncol = 3)
mosquito_proportions <- mosquito_proportions / rowSums(mosquito_proportions)

seasonality <- seasonality[sample(nrow(seasonality), n, replace = TRUE), ]

r_int <- lhs::randomLHS(n, period * 3)

# nets
llins <- lapply(seq(n), function(i) qunif(r_int[i, seq(period)], min=0, max=0.8))
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
irs <- lapply(seq(n), function(i) qunif(r_int[i, seq(period) + period], min=0, max=0.85))

# tx
tx <- lapply(seq(n), function(i) qunif(r_int[i, seq(period) + period * 2], min=0, max=0.85))

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
    prevalence_rendering_min_ages = c(2 * year),
    prevalence_rendering_max_ages = c(10 * year),
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

  # tx
  parameters <- malariasimulation::set_clinical_treatment(
    parameters,
    drug = 2,
    timesteps = one_round_timesteps + warmup * year,
    coverages = tx[[i]]
  )

  print(paste('row', i))

  malariasimulation::run_simulation(
    (period + warmup) * year,
    parameters=parameters
  )[c(
    'n_inc_clinical_0_36500',
    'n_0_36500',
    'n_detect_730_3650',
    'n_730_3650',
    'EIR_gamb',
    'EIR_arab',
    'EIR_fun'
  )]
}

batches <- split(
  seq(nrow(params)),
  (seq(nrow(params))-1) %/% batch_size
)

get_EIR <- function(result) {
  colSums(rbind(result$EIR_gamb, result$EIR_fun, result$EIR_arab))
}

daily_parameters <- function(i, result) {
  species_vector <- unlist(lapply(
    mosquito_params[[i]],
    function(p) {
      c(
        p$Q0,
        p$phi_indoors,
        p$phi_bednets
      )
    }
  ))
  row <- params[i,]
  c(
    mean(get_EIR(result)[seq((warmup - 1)*year, (warmup)*year)]),
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
    c(llins[[i]], irs[[i]], tx[[i]]),
    ncol = 3,
    nrow = period
  )
}

daily_outputs <- function(result) {
  result <- result[seq(warmup*year + 1, nrow(result)),]
  matrix(
    c(
      result$n_detect_730_3650 / result$n_730_3650,
      result$n_inc_clinical_0_36500 / result$n_0_36500,
      get_EIR(result)
    ),
    ncol = 3,
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
