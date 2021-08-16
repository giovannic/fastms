args = commandArgs(trailingOnly=TRUE)
node <- as.numeric(args[1])
global_fits <- args[2]
batch_size <- as.numeric(args[3])
n_batches <- as.numeric(args[4])
seed <- as.numeric(args[5])
sim_length <- as.numeric(args[6])
out_dir <- args[7]

print(paste0('beginning node ', node))

fits <- readRDS(global_fits)

set.seed(seed)
n <- n_batches * batch_size
seasonality <- do.call(rbind, fits$seasonality)
params <- data.frame(
  average_age = runif(n, 20 * 365, 40 * 365),
  init_EIR = runif(n, 0, 100),
  Q0 = runif(n, 0, 1)
)
params <- cbind(
  params,
  seasonality[sample(nrow(seasonality), n, replace = TRUE), ]
)

process_row <- function(row) {
  parameters <- malariasimulation::get_parameters(list(
    human_population = 10000,
    individual_mosquitoes = FALSE,
    model_seasonality = TRUE,
    g0 = row$seasonal_a0,
    g = c(row$seasonal_a1, row$seasonal_a2, row$seasonal_a3),
    h = c(row$seasonal_b1, row$seasonal_b2, row$seasonal_b3),
    Q0 = row$Q0,
    average_age = row$average_age
  ))
  parameters <- malariasimulation::set_equilibrium(parameters, row$init_EIR)

  malariasimulation::run_simulation(
    sim_length,
    parameters=parameters
  )[c('pv_730_3650', 'EIR')]
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
    results <- lapply(
      batches[[batch_i]],
      function (i) process_row(params[i,])
    )

    outdata <- params[batches[[batch_i]],]
    outdata$prev <- lapply(results, function(x) x$pv_730_3650)
    outdata$eir <- lapply(results, function(x) x$EIR)

    jsonlite::write_json(outdata, outpath, pretty=TRUE)
    print(paste0('node ', node, ' batch ', batch_i, ' completed'))
    print(Sys.time())
    print(Sys.time() - start_time)
  }
}
