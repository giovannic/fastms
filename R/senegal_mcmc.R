args = commandArgs(trailingOnly=TRUE)
node <- as.numeric(args[1])
batch_size <- as.numeric(args[2])
n_batches <- as.numeric(args[3])
seed <- as.numeric(args[4])
warmup <- as.numeric(args[5])
yearly_dir <- args[6]

print(paste0('beginning node ', node))

history_years <- 16
period <- history_years
year <- 365
month <- 30
population <- 10000

set.seed(seed)

db <- 10
dc <- 30
a0 <- 2920
rho<- 0.85
immunity_scale <- function(a, d) {
  (1 + rho / (d / a0 - 1) * exp(-a /a0) - (1 + rho/(d / a0 + 1)) * exp(-a / d))
}

n <- n_batches * batch_size

r <- lhs::randomLHS(n, 14)
n_params <- 13

params <- data.frame(
  average_age = qunif(r[,1], min=20 * 365, max=40 * 365),
  init_EIR = qunif(r[,2], min=0, max=100),
  sigma_squared = qunif(r[,3], min=1, max=3),
  du = qunif(r[,4], min=50, max=150),
  kb = qunif(r[,5], min=0.01, max=10),
  ub = qunif(r[,6], min=1, max=10),
  uc = qunif(r[,7], min=1, max=10), # not documented
  ud = qunif(r[,8], min=1, max=10), # not documented
  kc = qunif(r[,9], min=0.01, max=10),
  b0 = qunif(r[,10], min=0.01, max=0.99),
  ct = qunif(r[,11], min=0, max=1),
  cd = qunif(r[,12], min=0, max=1),
  #ca = runif(n, 0, 1), #TODO: how is this changed?
  cu = qunif(r[,13], min=0, max=1)
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

params$b1 <- qunif(r[,14], min=0, max=1) * params$b0

species <- c('gamb', 'arab', 'fun')
Q0 <- c(.92, .71, .94)
Q0_in <- c(.97, .96, .98)
Q0_bed <- c(.89, .9, .9)
mosquito_params <- lapply(
  seq_along(species),
  function(j) {
    list(
      species = species[[j]],
      mum = 0.1253333,
      blood_meal_rates = 1 / 3,
      foraging_time = 0.69,
      Q0 = Q0[[j]],
      phi_indoors = Q0_in[[j]],
      phi_bednets = Q0_bed[[j]]
    )
  }
)
mosquito_proportions <- c(0, 0.1481481, 0.8518519)

seasonality <- list(
  seasonal_a0 = 0.1555295,
  seasonal_a1 = -0.1446177,
  seasonal_b1 = -0.2020749,
  seasonal_a2 = -0.03262707,
  seasonal_b2 = 0.1604087,
  seasonal_a3 = 0.07101577,
  seasonal_b3 = -0.02115616
)

# nets
llins <- c(0.0000, 0.0000, 0.0000, 0.0000, 0.0528, 0.0955, 0.1213, 0.1465, 0.1872, 0.2876, 0.2816, 0.2670,
  0.3272, 0.4384, 0.5704, 0.6074)

# irs
irs <- c(0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
  0.05693862, 0.05263740, 0.05249131, 0.07337312, 0.06648029, 0.07970626, 0.04879426,
  0.06499360, 0.06499360)

# tx
tx <- c(0.0000, 0.0000, 0.0000, 0.0000, 0.0126, 0.0252, 0.0448, 0.0528, 0.0460, 0.0318, 0.0340, 0.0216,
0.0130, 0.0158, 0.0062, 0.0202)

one_round_timesteps <- seq(0, period - 1) * year

process_row <- function(i) {
  row <- params[i,]
  seas_row <- seasonality
  parameters <- malariasimulation::get_parameters(list(
    human_population = population,
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
    mosquito_params,
    proportions = mosquito_proportions
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
    coverages = llins,
    retention = 5 * year,
    dn0 = matrix(.533, nrow=period, ncol=3),
    rn = matrix(.56, nrow=period, ncol=3),
    rnm = matrix(.24, nrow=period, ncol=3),
    gamman = rep(2.64 * year, period)
  )

  # spraying
  parameters <- malariasimulation::set_spraying(
    parameters,
    timesteps = one_round_timesteps + (warmup * year),
    coverages = irs,
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
    coverages = tx
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
  colSums(rbind(
    result$EIR_gamb,
    result$EIR_fun,
    result$EIR_arab
  )) / population * year
}

daily_parameters <- function(i, result) {
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
    row$cu
  )
}

yearly_parameters <- function(i, result) {
  daily_parameters(i, result)
}

yearly_outputs <- function(result) {
  result <- result[seq(warmup*year + 1, nrow(result)),]
  series <- result$n_detect_730_3650 / result$n_730_3650
  t(matrix(series, nrow=year))
}

format_yearly <- function(i, result) {
  list(
    parameters = yearly_parameters(i, result),
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
