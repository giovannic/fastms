FROM rocker/r-ver:4.0.3

RUN R -e "install.packages(c('remotes', 'lhs'))"
RUN R -e "remotes::install_github('mrc-ide/malariasimulation@17311c0')"
